//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// ~span() = default;

#include <span>
#include <type_traits>

template <class T>
constexpr void testDestructor() {
  static_assert(std::is_nothrow_destructible_v<T>);
  static_assert(std::is_trivially_destructible_v<T>);
}

void test() {
  testDestructor<std::span<int, 1>>();
  testDestructor<std::span<int>>();
}
