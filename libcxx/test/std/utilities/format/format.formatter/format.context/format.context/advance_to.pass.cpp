//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// void advance_to(iterator it);

#include <format>
#include <cassert>
#include <string>

#include "test_macros.h"
#include "test_format_context.h"
#include "test_iterators.h"

template <class OutIt, class CharT>
void test(
    std::basic_format_args<std::basic_format_context<OutIt, CharT>> args) {
  {
    std::basic_string<CharT> str[3] = {
        std::basic_string<CharT>{1}, std::basic_string<CharT>{1}, std::basic_string<CharT>{1}};
    std::basic_format_context context = test_format_context_create(OutIt{str[0].begin()}, args);
    *context.out()                    = CharT('a');
    context.advance_to(OutIt{str[1].begin()});
    *context.out() = CharT('b');
    context.advance_to(OutIt{str[2].begin()});
    *context.out() = CharT('c');

    assert(str[0].size() == 1);
    assert(str[0].front() == CharT('a'));
    assert(str[1].size() == 1);
    assert(str[1].front() == CharT('b'));
    assert(str[2].size() == 1);
    assert(str[2].front() == CharT('c'));
  }
}

void test() {
  test(std::basic_format_args(
      std::make_format_args<
          std::basic_format_context<cpp20_output_iterator<std::basic_string<char>::iterator>, char>>()));

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test(std::basic_format_args(
      std::make_format_args<
          std::basic_format_context<cpp20_output_iterator<std::basic_string<wchar_t>::iterator>, wchar_t>>()));
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test(std::basic_format_args(
      std::make_format_args<
          std::basic_format_context<cpp20_output_iterator<std::basic_string<char8_t>::iterator>, char8_t>>()));
#endif
  test(std::basic_format_args(
      std::make_format_args<
          std::basic_format_context<cpp20_output_iterator<std::basic_string<char16_t>::iterator>, char16_t>>()));
  test(std::basic_format_args(
      std::make_format_args<
          std::basic_format_context<cpp20_output_iterator<std::basic_string<char32_t>::iterator>, char32_t>>()));
}

int main(int, char**) {
  test();

  return 0;
}
