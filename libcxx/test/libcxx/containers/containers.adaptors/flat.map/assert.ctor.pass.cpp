//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <flat_map>

// flat_map(key_container_type , mapped_container_type , const key_compare& __comp = key_compare())
//

#include <flat_map>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  using M = std::flat_map<int, int>;

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] { M m{{1, 2, 3}, {4}}; }()), "flat_map keys and mapped containers have different size");

  return 0;
}
