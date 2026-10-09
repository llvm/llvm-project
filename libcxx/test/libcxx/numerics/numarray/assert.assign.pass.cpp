//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// Test that a proxy array assigned to a valarray must have the same size as the valarray.

// REQUIRES: can-test-hardening-assertions-fast

#include <cstddef>
#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  std::valarray<int> source(1, 8);
  std::valarray<int> v(3);

  // Proxy arrays with 2 elements.
  std::valarray<std::size_t> gslice_size(2, 1);
  std::valarray<std::size_t> gslice_stride(1, 1);
  std::valarray<bool> mask(false, 8);
  mask[std::slice(0, 2, 1)] = true;
  std::size_t index_array[] = {0, 1};
  std::valarray<std::size_t> indices(index_array, 2);

  v = source[std::slice(0, 3, 1)]; // Check that there's no assertion for arrays of the same size.

  TEST_LIBCPP_ASSERT_FAILURE(v = source[std::slice(0, 2, 1)], "valarray::operator=(slice_array) size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(
      v = source[std::gslice(0, gslice_size, gslice_stride)], "valarray::operator=(gslice_array) size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v = source[mask], "valarray::operator=(mask_array) size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v = source[indices], "valarray::operator=(indirect_array) size mismatch");

  return 0;
}
