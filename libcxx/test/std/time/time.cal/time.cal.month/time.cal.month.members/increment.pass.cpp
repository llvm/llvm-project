//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class month;

//  constexpr month& operator++() noexcept;
//  constexpr month operator++(int) noexcept;

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

constexpr bool test() {
  using month = std::chrono::month;
  for (unsigned i = 0; i <= 15; ++i) {
    month m1(i);
    month m2 = m1++;
    assert(m1.ok());
    assert(m1 != m2);

    unsigned exp = i + 1;
    while (exp > 12)
      exp -= 12;
    assert(static_cast<unsigned>(m1) == exp);
  }
  for (unsigned i = 0; i <= 15; ++i) {
    month m1(i);
    month m2 = ++m1;
    assert(m1.ok());
    assert(m2.ok());
    assert(m1 == m2);

    unsigned exp = i + 1;
    while (exp > 12)
      exp -= 12;
    assert(static_cast<unsigned>(m1) == exp);
  }

  return true;
}

int main(int, char**) {
  using month = std::chrono::month;

  ASSERT_NOEXCEPT(++(std::declval<month&>()));
  ASSERT_NOEXCEPT((std::declval<month&>())++);

  ASSERT_SAME_TYPE(month, decltype(std::declval<month&>()++));
  ASSERT_SAME_TYPE(month&, decltype(++std::declval<month&>()));

  test();
  static_assert(test());

  return 0;
}
