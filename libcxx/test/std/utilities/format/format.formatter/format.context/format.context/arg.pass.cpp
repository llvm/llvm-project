//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11

// <format>

// basic_format_arg<basic_format_context> arg(size_t id) const;

#include <format>
#include <cassert>

#include "test_basic_format_arg.h"
#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

template <class OutIt, class CharT>
void test() {
  std::basic_string<CharT> string = MAKE_STRING(CharT, "string");
  auto store = std::make_format_args<std::basic_format_context<OutIt, CharT>>(
      true, CharT('a'), 42, string);
  std::basic_format_args args = store;

  std::basic_string<CharT> output;
  const std::basic_format_context context =
      test_format_context_create(OutIt{output}, args);
  LIBCPP_ASSERT(args.__size() == 4);
  for (size_t i = 0, e = args.__size(); i != e; ++i) {
    assert(context.arg(i));
  }
  assert(!context.arg(args.__size()));

  assert(test_basic_format_arg(context.arg(0), true));
  assert(test_basic_format_arg(context.arg(1), CharT('a')));
  assert(test_basic_format_arg(context.arg(2), 42));
  assert(test_basic_format_arg(context.arg(3),
                               std::basic_string_view<CharT>(string)));
}

int main(int, char**) {
  test<std::back_insert_iterator<std::basic_string<char>>, char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::back_insert_iterator<std::basic_string<wchar_t>>, wchar_t>();
#endif
#ifndef _LIBCPP_HAS_NO_CHAR8_T
  test<std::back_insert_iterator<std::basic_string<char8_t>>, char8_t>();
#endif
#ifndef TEST_HAS_NO_UNICODE_CHARS
  test<std::back_insert_iterator<std::basic_string<char16_t>>, char16_t>();
  test<std::back_insert_iterator<std::basic_string<char32_t>>, char32_t>();
#endif

  return 0;
}
