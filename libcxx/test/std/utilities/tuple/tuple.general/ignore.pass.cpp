//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <tuple>

// inline constexpr ignore-type ignore;

#include <cassert>
#include <tuple>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_trivial<decltype(std::ignore)>::value, "");

void test() {
  {
    [[maybe_unused]] constexpr auto& ignore_v = std::ignore;
  }
  { // Test that std::ignore provides constexpr converting assignment.
    constexpr auto& res = (std::ignore = 42);
    static_assert(noexcept(res = (std::ignore = 42)), "Must be noexcept");
    assert(&res == &std::ignore);
  }
  { // Test that std::ignore provides constexpr copy/move constructors
    constexpr auto copy  = std::ignore;
    [[maybe_unused]] constexpr auto moved = std::move(copy);
  }
  { // Test that std::ignore provides constexpr copy/move assignment
    constexpr auto copy  = std::ignore;
    copy                 = std::ignore;
    constexpr auto moved = std::ignore;
    moved                = std::move(copy);
  }
}

constexpr bool test_constexpr() {
#if TEST_STD_VER >= 14
  {
    auto& ignore_v = std::ignore;
    ((void)ignore_v);
  }
  { // Test that std::ignore provides constexpr converting assignment.
    auto& res = (std::ignore = 42);
    static_assert(noexcept(res = (std::ignore = 42)), "Must be noexcept");
    assert(&res == &std::ignore);
  }
  { // Test that std::ignore provides constexpr copy/move constructors
    auto copy  = (std::ignore = 42);
    auto moved = std::move(copy);
    ((void)moved);
  }
  { // Test that std::ignore provides constexpr copy/move assignment
    auto copy  = (std::ignore = 82);
    copy       = std::ignore;
    auto moved = (std::ignore = 94);
    moved      = std::move(copy);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test_constexpr(), "");

  return 0;
}
