//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <charconv>

// struct to_chars_result
//   constexpr explicit operator bool() const noexcept { return ec == errc{}; }

#include <charconv>

#include <cassert>
#include <system_error>
#include <type_traits>

#include "test_macros.h"

static_assert(!std::is_convertible_v<std::to_chars_result, bool>);
static_assert(std::is_constructible_v<bool, std::to_chars_result>);

constexpr bool test() {
  // True
  {
    std::to_chars_result value{nullptr, std::errc{}};
    assert(bool(value) == true);
    static_assert(noexcept(bool(value)) == true);
  }
  // False
  {
    std::to_chars_result value{nullptr, std::errc::value_too_large};
    assert(bool(value) == false);
    static_assert(noexcept(bool(value)) == true);
  }

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  return 0;
}
