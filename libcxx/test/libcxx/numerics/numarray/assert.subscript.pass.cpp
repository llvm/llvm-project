//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast

// <valarray>

// template<class T> class valarray;

// slice_array<value_type> operator[](slice s);
// gslice_array<value_type> operator[](const gslice& gs);
// gslice_array<value_type> operator[](gslice&& gs);
// mask_array<value_type> operator[](const valarray<bool>& vb);
// mask_array<value_type> operator[](valarray<bool>&& vb);
// indirect_array<value_type> operator[](const valarray<size_t>& vs);
// indirect_array<value_type> operator[](valarray<size_t>&& vs);

// Test that the selected elements must exist.

#include <cstddef>
#include <limits>
#include <valarray>

#include "check_assertion.h"

std::gslice make_gslice(std::size_t start, std::size_t len0, std::size_t len1, std::size_t str0, std::size_t str1) {
  std::size_t lengths[] = {len0, len1};
  std::size_t strides[] = {str0, str1};
  return std::gslice(start, std::valarray<std::size_t>(lengths, 2), std::valarray<std::size_t>(strides, 2));
}

int main(int, char**) {
  std::valarray<int> array(10);

  { // slice
    // Check that there's no assertion on valid slices.
    (void)array[std::slice(0, 10, 1)];
    (void)array[std::slice(1, 5, 2)];
    (void)array[std::slice(9, 3, 0)];
    (void)array[std::slice(20, 0, 1)];

    TEST_LIBCPP_ASSERT_FAILURE(
        array[std::slice(0, 11, 1)], "valarray::operator[](slice) selects elements out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(
        array[std::slice(10, 1, 1)], "valarray::operator[](slice) selects elements out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(
        array[std::slice(4, 4, 2)], "valarray::operator[](slice) selects elements out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(
        array[std::slice(10, 3, 0)], "valarray::operator[](slice) selects elements out of bounds");

    // (size() - 1) * stride() wraps around to 0, but the second element is far out of bounds.
    const std::size_t half = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 1);
    TEST_LIBCPP_ASSERT_FAILURE(
        array[std::slice(0, 3, half)], "valarray::operator[](slice) selects elements out of bounds");
  }

  { // gslice
    // Check that there's no assertion on valid generalized slices.
    const std::gslice all = make_gslice(0, 2, 5, 5, 1); // 0 to 9
    (void)array[all];
    (void)array[make_gslice(3, 2, 2, 5, 1)];  // 3, 4, 8, 9
    (void)array[make_gslice(9, 3, 2, 0, 0)];  // 9 six times
    (void)array[make_gslice(20, 0, 2, 1, 1)]; // empty

    const std::gslice past_end = make_gslice(6, 2, 2, 4, 1); // 6, 7, 10, 11
    TEST_LIBCPP_ASSERT_FAILURE(array[past_end], "valarray::operator[](gslice) selects elements out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(
        array[make_gslice(1, 2, 5, 5, 1)], "valarray::operator[](gslice) selects elements out of bounds"); // 1 to 10
  }

  { // valarray<bool>
    // Check that there's no assertion on valid masks, including masks of a different size that select no element
    // past the end.
    const std::valarray<bool> all(true, 10);
    (void)array[all];
    (void)array[std::valarray<bool>(true, 4)];
    (void)array[std::valarray<bool>(false, 20)];
    std::valarray<bool> longer(false, 12);
    longer[9] = true;
    (void)array[longer];

    std::valarray<bool> one_past(false, 11);
    one_past[10] = true;
    TEST_LIBCPP_ASSERT_FAILURE(array[one_past], "valarray::operator[](valarray<bool>) selects elements out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(
        array[std::valarray<bool>(true, 12)], "valarray::operator[](valarray<bool>) selects elements out of bounds");

    std::valarray<int> empty;
    TEST_LIBCPP_ASSERT_FAILURE(
        empty[std::valarray<bool>(true, 1)], "valarray::operator[](valarray<bool>) selects elements out of bounds");
  }

  { // valarray<size_t>
    // Check that there's no assertion on valid indices.
    std::size_t valid[] = {0, 9, 9, 3};
    const std::valarray<std::size_t> indices(valid, 4);
    (void)array[indices];
    (void)array[std::valarray<std::size_t>()];

    std::size_t past_end[] = {0, 9, 10, 3};
    const std::valarray<std::size_t> past_end_indices(past_end, 4);
    TEST_LIBCPP_ASSERT_FAILURE(
        array[past_end_indices], "valarray::operator[](valarray<size_t>) selects elements out of bounds");
    std::size_t max[] = {static_cast<std::size_t>(-1)};
    TEST_LIBCPP_ASSERT_FAILURE(array[std::valarray<std::size_t>(max, 1)],
                               "valarray::operator[](valarray<size_t>) selects elements out of bounds");
  }

  return 0;
}
