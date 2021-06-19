//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <format>

// constexpr end() const noexcept;

#include <format>

#include <cassert>
#include <string_view>

#include "test_macros.h"

template <class CharT>
constexpr void test(const CharT* fmt) {
  {
    std::basic_format_parse_context<CharT> context(fmt);
    assert(context.end() == &fmt[3]);
    ASSERT_NOEXCEPT(context.end());
  }
  {
    std::basic_string_view view{fmt};
    std::basic_format_parse_context context(view);
    assert(context.end() == view.end());
    ASSERT_NOEXCEPT(context.end());
  }
}

constexpr bool test() {
  test("abc");
  test(L"abc");
#ifndef _LIBCPP_HAS_NO_CHAR8_T
  test(u8"abc");
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
  test(u"abc");
  test(U"abc");
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
