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

TEST_CONSTEXPR_CXX14 bool test() {
  { [[maybe_unused]] auto& ignore_v = std::ignore; }

  { // Test that std::ignore provides converting assignment.
    auto& res = (std::ignore = 42);
    static_assert(noexcept(res = (std::ignore = 42)), "Must be noexcept");
    assert(&res == &std::ignore);
  }
  { // Test bit-field binding.
    struct S {
      unsigned int bf : 3;
    };
    S s{0b010};
    auto& res = (std::ignore = s.bf);
    assert(&res == &std::ignore);
  }
  { // Test that std::ignore provides copy/move constructors
    auto copy                   = std::ignore;
    [[maybe_unused]] auto moved = std::move(copy);
  }
  { // Test that std::ignore provides copy/move assignment
    auto copy  = std::ignore;
    copy       = std::ignore;
    auto moved = std::ignore;
    moved      = std::move(copy);
  }

#if TEST_STD_VER >= 17
  { std::ignore = test_nodiscard(); }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif

  return 0;
}
