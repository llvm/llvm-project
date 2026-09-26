//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers, std-at-least-c++26
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <inplace_vector>

#include <inplace_vector>

#include "check_assertion.h"

int main(int, char**) {
  std::inplace_vector<int, 2> v{1, 2};

  TEST_LIBCPP_ASSERT_FAILURE(
      v.erase(v.end()), "inplace_vector<T,N>::erase(iterator) called with a non-dereferenceable iterator");
  TEST_LIBCPP_ASSERT_FAILURE(
      v.erase(v.end(), v.begin()), "inplace_vector<T,N>::erase(first, last) called with invalid range");

  return 0;
}
