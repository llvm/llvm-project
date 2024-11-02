//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// explicit operator bool() const noexcept
//
// Note more testing is done in the unit test for:
// template<class Visitor, class Context>
//   see below visit_format_arg(Visitor&& vis, basic_format_arg<Context> arg);

#include <format>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <class CharT>
void test() {
  using Context = std::basic_format_context<CharT*, CharT>;
  {
    std::basic_format_arg<Context> format_arg{};
    ASSERT_NOEXCEPT(!format_arg);
    assert(!format_arg);
    ASSERT_NOEXCEPT(static_cast<bool>(format_arg));
    assert(!static_cast<bool>(format_arg));
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
