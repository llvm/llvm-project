//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::istream<T>

#include <cassert>
#include <ranges>
#include <sstream>
#include <type_traits>

#include "test_macros.h"
#include "utils.h"

static_assert(!std::is_invocable_v<decltype((std::views::istream<int>))>);
static_assert(std::is_invocable_v<decltype((std::views::istream<int>)), std::istream&>);
static_assert(!std::is_invocable_v<decltype((std::views::istream<int>)), const std::istream&>);
static_assert(!std::is_invocable_v<decltype((std::views::istream<int>)), int>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_invocable_v<decltype((std::views::istream<int>)), std::wistream&>);
static_assert(!std::is_invocable_v<decltype((std::views::istream<int>)), const std::wistream&>);
#endif

template <class CharT>
void test() {
  auto iss = make_string_stream<CharT>("12   3");
  auto isv = std::views::istream<int>(iss);
  auto it  = isv.begin();
  assert(*it == 12);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
