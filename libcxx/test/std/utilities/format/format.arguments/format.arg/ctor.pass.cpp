//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// basic_format_arg() noexcept;

// The class has several exposition only private constructors. These are tested
// in visit_format_arg.pass.cpp

#include <format>
#include <cassert>

#include "test_macros.h"

template <class CharT>
void test() {
  using Context = std::basic_format_context<CharT*, CharT>;

  ASSERT_NOEXCEPT(std::basic_format_arg<Context>{});

  std::basic_format_arg<Context> format_arg{};
  assert(!format_arg);
}

void test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test<char8_t>();
#endif
  test<char16_t>();
  test<char32_t>();
}

int main(int, char**) {
  test();

  return 0;
}
