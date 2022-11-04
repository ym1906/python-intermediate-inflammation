"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0], [0, 0], [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2], [3, 4], [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_max():
    """Test that the max function works for an array of positive integers"""
    from inflammation.models import daily_max

    test_input = np.array([[1, 2, 3], [9, 3, 1], [4, 1, 8]])
    test_result = np.array([9, 3, 8])
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_min():
    """Test that the min function works for an array of positive and negative integers"""
    from inflammation.models import daily_min

    test_input = np.array([[4, -2, 5], [1, -6, 2], [-4, -9, 2]])
    test_result = np.array([-4, -9, 2])
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([["Hi", "Bye"], ["Cats", "Dogs"]])
