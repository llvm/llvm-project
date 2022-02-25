//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// constexpr explicit
// basic_format_parse_context(basic_string_view<charT> fmt,
//                            size_t num_args = 0) noexcept

#include <format>

#include <cassert>
#include <string_view>
#include <type_traits>

#include "test_macros.h"

template <class CharT>
constexpr void test(const CharT* fmt) {
  // Validate the constructor is explicit.
  static_assert(
      !std::is_convertible_v<std::basic_string_view<CharT>,
                             std::basic_format_parse_context<CharT> >);
  static_assert(
      !std::is_copy_constructible_v<std::basic_format_parse_context<CharT> >);
  static_assert(
      !std::is_copy_assignable_v<std::basic_format_parse_context<CharT> >);
  // The move operations are implicitly deleted due to the
  // deleted copy operations.
  static_assert(
      !std::is_move_constructible_v<std::basic_format_parse_context<CharT> >);
  static_assert(
      !std::is_move_assignable_v<std::basic_format_parse_context<CharT> >);

  ASSERT_NOEXCEPT(
      std::basic_format_parse_context{std::basic_string_view<CharT>{}});
  ASSERT_NOEXCEPT(
      std::basic_format_parse_context{std::basic_string_view<CharT>{}, 42});

  {
    std::basic_format_parse_context<CharT> context(fmt);
    assert(std::to_address(context.begin()) == &fmt[0]);
    assert(std::to_address(context.end()) == &fmt[3]);
  }
  {
    std::basic_string_view view{fmt};
    std::basic_format_parse_context context(view);
    assert(context.begin() == view.begin());
    assert(context.end() == view.end());
  }
}

constexpr bool test() {
  test("abc");
  test(L"abc");
#ifndef TEST_HAS_NO_CHAR8_T
  test(u8"abc");
#endif
#ifndef TEST_HAS_NO_UNICODE_CHARS
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
