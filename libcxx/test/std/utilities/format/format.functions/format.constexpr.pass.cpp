//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <format>

// template<class... Args>
//   constexpr string format(format-string<Args...> fmt, Args&&... args);
// template<class... Args>
//   constexpr wstring format(wformat-string<Args...> fmt, Args&&... args);

#include <cassert>
#include <format>
#include <string>
#include <string_view>

#include "test_macros.h"

constexpr bool test() {
  // No replacement fields
  assert(std::format("hello") == "hello");

  // Integers
  // assert(std::format("{}", 42) == "42");
  // assert(std::format("{}", -42) == "-42");
  // assert(std::format("{:x}", 255u) == "ff");
  // assert(std::format("{:>5}", 42) == "   42");

  // bool and char
  // assert(std::format("{}", true) == "true");
  // assert(std::format("{}", 'a') == "a");

  // Strings
  // assert(std::format("{}", "abc") == "abc");
  //  assert(std::format("{}", std::string_view{"abc"}) == "abc");
  //  assert(std::format("{:*^7}", "abc") == "**abc**");

  // Several arguments
  // assert(std::format("{} + {} = {}", 1, 2, 3) == "1 + 2 = 3");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // assert(std::format(L"{}", 42) == L"42");
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
