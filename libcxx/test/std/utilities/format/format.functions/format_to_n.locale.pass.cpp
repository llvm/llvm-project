//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11
// TODO FMT Investigate AppleClang ICE
// UNSUPPORTED: apple-clang-13

// <format>

// template<class Out, class... Args>
//   format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                       const locale& loc, string_view fmt,
//                                       const Args&... args);
// template<class Out, class... Args>
//   format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                       const locale& loc, wstring_view fmt,
//                                       const Args&... args);

#include <format>
#include <algorithm>
#include <cassert>
#include <list>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"
#include "string_literal.h"

auto test = []<string_literal fmt, class CharT, class... Args>(std::basic_string_view<CharT> expected,
                                                               const Args&... args) constexpr {
  {
    std::list<CharT> out;
    std::format_to_n_result result =
        std::format_to_n(std::back_inserter(out), 0, std::locale(), fmt.template sv<CharT>(), args...);
    // To avoid signedness warnings make sure formatted_size uses the same type
    // as result.size.
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(std::locale(), fmt.template sv<CharT>(), args...);

    assert(result.size == formatted_size);
    assert(out.empty());
  }
  {
    std::vector<CharT> out;
    std::format_to_n_result result =
        std::format_to_n(std::back_inserter(out), 5, std::locale(), fmt.template sv<CharT>(), args...);
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(std::locale(), fmt.template sv<CharT>(), args...);
    diff_type size = std::min<diff_type>(5, formatted_size);

    assert(result.size == formatted_size);
    assert(std::equal(out.begin(), out.end(), expected.begin(), expected.begin() + size));
  }
  {
    std::basic_string<CharT> out;
    std::format_to_n_result result =
        std::format_to_n(std::back_inserter(out), 1000, std::locale(), fmt.template sv<CharT>(), args...);
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(std::locale(), fmt.template sv<CharT>(), args...);
    diff_type size = std::min<diff_type>(1000, formatted_size);

    assert(result.size == formatted_size);
    assert(out == expected.substr(0, size));
  }
  {
    // Test the returned iterator.
    std::basic_string<CharT> out(10, CharT(' '));
    std::format_to_n_result result =
        std::format_to_n(out.begin(), 10, std::locale(), fmt.template sv<CharT>(), args...);
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(std::locale(), fmt.template sv<CharT>(), args...);
    diff_type size = std::min<diff_type>(10, formatted_size);

    assert(result.size == formatted_size);
    assert(result.out == out.begin() + size);
    assert(out.substr(0, size) == expected.substr(0, size));
  }
  {
    static_assert(std::is_signed_v<std::iter_difference_t<CharT*>>,
                  "If the difference type isn't negative the test will fail "
                  "due to using a large positive value.");
    CharT buffer[1] = {CharT(0)};
    std::format_to_n_result result = std::format_to_n(buffer, -1, std::locale(), fmt.template sv<CharT>(), args...);
    using diff_type = decltype(result.size);
    diff_type formatted_size = std::formatted_size(std::locale(), fmt.template sv<CharT>(), args...);

    assert(result.size == formatted_size);
    assert(result.out == buffer);
    assert(buffer[0] == CharT(0));
  }
};

auto test_exception = []<class CharT, class... Args>(std::string_view what, std::basic_string_view<CharT> fmt,
                                                     const Args&... args) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    std::basic_string<CharT> out;
    std::format_to_n(std::back_inserter(out), 0, std::locale(), fmt, args...);
    assert(false);
  } catch (const std::format_error& e) {
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
  format_tests<char>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_char_to_wchar_t(test);
  format_tests<wchar_t>(test, test_exception);
#endif

  return 0;
}
