//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <valarray>

// template<class T> class slice_array;

// T __get(size_t i); // where i is out of bounds

#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  unsigned input[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const unsigned N = sizeof(input) / sizeof(input[0]);

  std::valarray<unsigned> array(input, N);

  {
    std::slice_array<unsigned> result = array[std::slice(0, 0, 0)];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(0), "slice_array.__get() index out of bounds");
  }
  {
    std::slice_array<unsigned> result = array[std::slice(0, N, 1)];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(N), "slice_array.__get() index out of bounds");
  }
  {
    std::slice_array<unsigned> result = array[std::slice(3, 2, 2)];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(2), "slice_array.__get() index out of bounds");
  }

  {
    std::slice_array<unsigned> result = array[std::slice(1, 3, 4)];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(3), "slice_array.__get() index out of bounds");
  }

  return 0;
}
