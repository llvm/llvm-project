//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast

// <valarray>

// Test that the argument of a valarray compound assignment must have the same size as the valarray.

#include <cstddef>
#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  std::valarray<int> v(1, 2);
  std::valarray<int> w(2, 3);

  v += std::valarray<int>(2, 2); // Check that there's no assertion for arrays of the same size.

  TEST_LIBCPP_ASSERT_FAILURE(v *= w, "valarray::operator*= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v /= w, "valarray::operator/= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v %= w, "valarray::operator%= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v += w, "valarray::operator+= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v -= w, "valarray::operator-= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v ^= w, "valarray::operator^= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v &= w, "valarray::operator&= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v |= w, "valarray::operator|= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v <<= w, "valarray::operator<<= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v >>= w, "valarray::operator>>= size mismatch");

  // The argument is an expression or one of the proxy arrays, each with 3 elements.
  std::valarray<int> source(1, 8);
  std::valarray<std::size_t> gslice_size(3, 1);
  std::valarray<std::size_t> gslice_stride(1, 1);
  std::valarray<bool> mask(false, 8);
  mask[std::slice(0, 3, 1)] = true;
  std::size_t index_array[] = {0, 1, 2};
  std::valarray<std::size_t> indices(index_array, 3);
  TEST_LIBCPP_ASSERT_FAILURE(v += w + w, "valarray::operator+= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v += source[std::slice(0, 3, 1)], "valarray::operator+= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(
      v += source[std::gslice(0, gslice_size, gslice_stride)], "valarray::operator+= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v += source[mask], "valarray::operator+= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(v += source[indices], "valarray::operator+= size mismatch");

  return 0;
}
