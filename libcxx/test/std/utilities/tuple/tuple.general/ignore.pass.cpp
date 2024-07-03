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

int main(int, char**) {
  static_assert(std::is_trivial<decltype(std::ignore)>::value, "");

  { [[maybe_unused]] constexpr auto& ignore_v = std::ignore; }
  { // Test that std::ignore provides constexpr converting assignment.
    constexpr auto& res = (std::ignore = 42);
    static_assert(noexcept(res = (std::ignore = 42)), "Must be noexcept");
    assert(&res == &std::ignore);
  }
  { // Test that std::ignore provides constexpr copy/move constructors
    constexpr auto copy                   = std::ignore;
    [[maybe_unused]] constexpr auto moved = std::move(copy);
  }
  { // Test that std::ignore provides constexpr copy/move assignment
    constexpr auto copy  = std::ignore;
    copy                 = std::ignore;
    constexpr auto moved = std::ignore;
    moved                = std::move(copy);
  }

  return 0;
}
