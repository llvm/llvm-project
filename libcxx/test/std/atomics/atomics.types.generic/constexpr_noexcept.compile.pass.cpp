//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-threads

#include <atomic>

#include "test_macros.h"

template <typename T>
constexpr bool test() {
  [[maybe_unused]] constexpr T a;
  static_assert(std::is_nothrow_constructible_v<T>);
  ASSERT_NOEXCEPT(T{});
  return true;
}

struct throwing {
  throwing() {}
};

struct trivial {
  int a;
};

void test() {
  static_assert(test<std::atomic<bool>>());
  static_assert(test<std::atomic<int>>());
  static_assert(test<std::atomic<int*>>());
  static_assert(test<std::atomic<trivial>>());
  static_assert(test<std::atomic_flag>());

  static_assert(!std::is_nothrow_constructible_v<std::atomic<throwing>>);
  ASSERT_NOT_NOEXCEPT(std::atomic<throwing>{});
}
