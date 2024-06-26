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

// template<class T> class mask_array;

// T __get(size_t i); // where i is out of bounds

#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  unsigned input[] = {0, 1, 2, 3, 4};
  const unsigned N = sizeof(input) / sizeof(input[0]);

  std::valarray<unsigned> array(input, N);

  {
    std::mask_array<unsigned> result = array[std::valarray<bool>(false, N)];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(0), "mask_array.__get() index out of bounds");
  }
  {
    std::mask_array<unsigned> result = array[std::valarray<bool>(true, N)];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(N), "mask_array.__get() index out of bounds");
  }

  {
    std::valarray<bool> mask(false, N);
    mask[1]                          = true;
    mask[3]                          = true;
    std::mask_array<unsigned> result = array[mask];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(2), "mask_array.__get() index out of bounds");
  }

  return 0;
}
