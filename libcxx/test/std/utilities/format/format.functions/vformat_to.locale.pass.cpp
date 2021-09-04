//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11

// <format>

// template<class Out>
//   Out vformat_to(Out out, const locale& loc, string_view fmt,
//                  format_args_t<type_identity_t<Out>, char> args);
// template<class Out>
//    Out vformat_to(Out out, const locale& loc, wstring_view fmt,
//                   format_args_t<type_identity_t<Out>, wchar_t> args);

#include <format>
#include <algorithm>
#include <cassert>
#include <list>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"

auto test = []<class CharT, class... Args>(std::basic_string<CharT> expected,
                                           std::basic_string<CharT> fmt,
                                           const Args&... args) {
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::vformat_to(
        out.begin(), std::locale(), fmt,
        std::make_format_args<std::basic_format_context<
            typename std::basic_string<CharT>::iterator, CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  {
    std::list<CharT> out;
    std::vformat_to(
        std::back_inserter(out), std::locale(), fmt,
        std::make_format_args<std::basic_format_context<
            std::back_insert_iterator<std::list<CharT>>, CharT>>(args...));
    assert(
        std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
  }
  {
    std::vector<CharT> out;
    std::vformat_to(
        std::back_inserter(out), std::locale(), fmt,
        std::make_format_args<std::basic_format_context<
            std::back_insert_iterator<std::vector<CharT>>, CharT>>(args...));
    assert(
        std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
  }
  {
    assert(expected.size() < 4096 && "Update the size of the buffer.");
    CharT out[4096];
    CharT* it = std::vformat_to(
        out, std::locale(), fmt,
        std::make_format_args<std::basic_format_context<CharT*, CharT>>(
            args...));
    assert(std::distance(out, it) == int(expected.size()));
    *it = '\0';
    assert(out == expected);
  }
};

auto test_exception = []<class CharT, class... Args>(
    std::string_view what, std::basic_string<CharT> fmt, const Args&... args) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    std::basic_string<CharT> out;
    std::vformat_to(
        std::back_inserter(out), std::locale(), fmt,
        std::make_format_args<std::basic_format_context<
            std::back_insert_iterator<std::basic_string<CharT>>, CharT>>(
            args...));
    assert(false);
  } catch (std::format_error& e) {
    LIBCPP_ASSERT(e.what() == what);
    return;
  }
  assert(false);
#else
  (void)what;
  (void)fmt;
  (void)sizeof...(args);
#endif
};

int main(int, char**) {
  format_tests_char_to_wchar_t(test);

  format_tests<char>(test, test_exception);
  format_tests<wchar_t>(test, test_exception);

  return 0;
}
