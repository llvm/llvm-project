//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast

// <valarray>

// template<class T> class slice_array;

// void operator=(const valarray<value_type>& v) const; // where v.size() differs
// const slice_array& operator=(const slice_array& sa) const; // where sa has a different size

#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  std::valarray<int> array(1, 16);
  std::slice_array<int> target = array[std::slice(0, 3, 2)];

  // Check that there's no assertion for arrays of the same size.
  target = std::valarray<int>(2, 3);
  target = array[std::slice(1, 3, 2)];

  TEST_LIBCPP_ASSERT_FAILURE(target = std::valarray<int>(2, 4), "slice_array::operator= size mismatch");
  TEST_LIBCPP_ASSERT_FAILURE(target = array[std::slice(1, 4, 2)], "slice_array::operator= size mismatch");

  return 0;
}
