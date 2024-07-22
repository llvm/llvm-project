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

// template<class T> class gslice_array;

// T __get(size_t i); // where i is out of bounds

#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  unsigned input[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const unsigned N = sizeof(input) / sizeof(input[0]);

  std::valarray<unsigned> array(input, N);

  {
    std::gslice_array<unsigned> result =
        array[std::gslice(0, std::valarray<std::size_t>(N, 1), std::valarray<std::size_t>(1, 1))];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(N), "gslice_array.__get() index out of bounds");
  }
  {
    std::valarray<std::size_t> sizes(2);
    sizes[0] = 2;
    sizes[1] = 3;

    std::valarray<std::size_t> strides(2);
    strides[0] = 6;
    strides[1] = 1;

    std::gslice_array<unsigned> result = array[std::gslice(1, sizes, strides)];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(6), "gslice_array.__get() index out of bounds");
  }

  return 0;
}
