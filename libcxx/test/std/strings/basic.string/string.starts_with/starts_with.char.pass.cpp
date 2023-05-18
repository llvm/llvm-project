//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <string>

// constexpr bool starts_with(charT x) const noexcept;

#include <string>
#include <cassert>

#include "test_macros.h"

template <class S>
constexpr void test_string() {
  S s1{};
  S s2{"abcde", 5};

  ASSERT_NOEXCEPT(s1.starts_with('e'));

  assert(!s1.starts_with('a'));
  assert(!s1.starts_with('x'));
  assert(s2.starts_with('a'));
  assert(!s2.starts_with('x'));
}

constexpr bool test() {
  test_string<std::string>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
