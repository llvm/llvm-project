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
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_trivial<decltype(std::ignore)>::value, "");

#if TEST_STD_VER >= 17
[[nodiscard]] constexpr int test_nodiscard() { return 8294; }
#endif

constexpr bool test() {
#if TEST_STD_VER >= 17
  { std::ignore = test_nodiscard(); }
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  { [[maybe_unused]] constexpr auto& ignore_v = std::ignore; }
  { // Test that std::ignore provides constexpr converting assignment.
    constexpr auto& res = (std::ignore = 42);
    static_assert(noexcept(res = (std::ignore = 42)), "Must be noexcept");
    assert(&res == &std::ignore);
  }
  { // Test bit-field binding.
    struct S {
      unsigned int bf : 3;
    };
    constexpr S s{0b010};
    constexpr auto& res = (std::ignore = s.bf);
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
