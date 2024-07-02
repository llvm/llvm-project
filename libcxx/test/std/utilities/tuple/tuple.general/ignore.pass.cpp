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

constexpr bool test() {
  {
    constexpr auto& ignore_v = std::ignore;
    ((void)ignore_v);
  }
  { // Test that std::ignore provides constexpr converting assignment.
    auto& res = (std::ignore = 42);
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

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");
  LIBCPP_STATIC_ASSERT(std::is_trivial<decltype(std::ignore)>::value, "");

  return 0;
}
