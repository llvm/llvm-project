//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast

// <valarray>

// template<class T> class gslice_array;

// const gslice_array& operator=(const gslice_array& ga) const; // where ga has a different size

#include <cstddef>
#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  std::valarray<int> array(1, 16);
  std::valarray<std::size_t> stride(2, 1);
  std::gslice_array<int> target = array[std::gslice(0, std::valarray<std::size_t>(3, 1), stride)];

  // Check that there's no assertion for arrays of the same size.
  target = array[std::gslice(1, std::valarray<std::size_t>(3, 1), stride)];

  TEST_LIBCPP_ASSERT_FAILURE(target = array[std::gslice(1, std::valarray<std::size_t>(4, 1), stride)],
                             "gslice_array::operator= size mismatch");

  return 0;
}
