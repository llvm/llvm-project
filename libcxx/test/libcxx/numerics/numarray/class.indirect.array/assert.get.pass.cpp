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

// template<class T> class indirect_array;

// T __get(size_t i); // where i is out of bounds

#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  unsigned input[] = {0, 1, 2, 3, 4};
  const unsigned N = sizeof(input) / sizeof(input[0]);

  std::valarray<unsigned> array(input, N);

  {
    std::indirect_array<unsigned> result = array[std::valarray<std::size_t>()];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(0), "indirect_array.__get() index out of bounds");
  }
  {
    std::indirect_array<unsigned> result = array[std::valarray<std::size_t>(std::size_t(0), std::size_t(N))];
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(N), "indirect_array.__get() index out of bounds");
  }

  {
    std::valarray<std::size_t> indirect(std::size_t(0), std::size_t(3));
    std::indirect_array<unsigned> result = array[indirect];
    indirect[0]                          = 4;
    indirect[1]                          = 1;
    indirect[2]                          = 3;
    TEST_LIBCPP_ASSERT_FAILURE(result.__get(3), "indirect_array.__get() index out of bounds");
  }

  return 0;
}
