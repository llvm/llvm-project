//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: availability-verbose_abort-missing

// <span>

// constexpr explicit(extent != dynamic_extent) span(std::initializer_list<value_type> il); // Since C++26

#include <cassert>
#include <span>
#include <initializer_list>

#include "check_assertion.h"

// template <std::size_t N>
// void test_runtime_assertion(std::span<const int, N> sp) {
//   static_assert(std::dynamic_extent != sp.extent);
// }

bool test() {
  TEST_LIBCPP_ASSERT_FAILURE(
      (std::span<const int, 4>({1, 2, 3, 9084, 5})), "Size mismatch in span's constructor _Extent != __il.size().");
  TEST_LIBCPP_ASSERT_FAILURE((std::span<const int, 4>(std::initializer_list<int>{1, 2, 3, 9084, 5})),
                             "Size mismatch in span's constructor _Extent != __il.size().");
  // TEST_LIBCPP_ASSERT_FAILURE((test_runtime_assertion(std::span<const int, 4>({1, 2, 3, 9084, 0}))),
  //                            "Size mismatch in span's constructor _Extent != __il.size().");

  return true;
}

int main(int, char**) {
  assert(test());

  return 0;
}
