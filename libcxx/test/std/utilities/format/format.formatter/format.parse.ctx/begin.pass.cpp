//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// constexpr begin() const noexcept;

#include <format>

#include <cassert>
#include <string_view>

#include "test_macros.h"

template <class CharT>
constexpr void test(const CharT* fmt) {
  {
    std::basic_format_parse_context<CharT> context(fmt);
    assert(std::to_address(context.begin()) == &fmt[0]);
    ASSERT_NOEXCEPT(context.begin());
  }
  {
    std::basic_string_view view{fmt};
    std::basic_format_parse_context context(view);
    assert(context.begin() == view.begin());
    ASSERT_NOEXCEPT(context.begin());
  }
}

constexpr bool test() {
  test("abc");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test(L"abc");
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test(u8"abc");
#endif
  test(u"abc");
  test(U"abc");

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
