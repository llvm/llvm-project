//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// <format>

// template<class Out>
//   Out vformat_to(Out out, const locale& loc, string_view fmt,
//                  format_args args);
// template<class Out>
//    Out vformat_to(Out out, const locale& loc, wstring_view fmt,
//                   wformat_args args);

#include <format>
#include <algorithm>
#include <cassert>
#include <list>
#include <vector>

#include "assert_macros.h"
#include "concat_macros.h"
#include "format_tests.h"
#include "string_literal.h"

auto test = []<class CharT, class... Args>(
                std::basic_string_view<CharT> expected, std::basic_string_view<CharT> fmt, Args&&... args) constexpr {
  {
    std::basic_string<CharT> out(expected.size(), CharT(' '));
    auto it = std::vformat_to(out.begin(), std::locale(), fmt, std::make_format_args<context_t<CharT>>(args...));
    assert(it == out.end());
    assert(out == expected);
  }
  {
    std::list<CharT> out;
    std::vformat_to(std::back_inserter(out), std::locale(), fmt, std::make_format_args<context_t<CharT>>(args...));
    assert(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
  }
  {
    std::vector<CharT> out;
    std::vformat_to(std::back_inserter(out), std::locale(), fmt, std::make_format_args<context_t<CharT>>(args...));
    assert(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
  }
  {
    assert(expected.size() < 4096 && "Update the size of the buffer.");
    CharT out[4096];
    CharT* it = std::vformat_to(out, std::locale(), fmt, std::make_format_args<context_t<CharT>>(args...));
    assert(std::distance(out, it) == int(expected.size()));
    // Convert to std::string since output contains '\0' for boolean tests.
    assert(std::basic_string<CharT>(out, it) == expected);
  }
};

auto test_exception = []<class CharT, class... Args>([[maybe_unused]] std::string_view what,
                                                     [[maybe_unused]] std::basic_string_view<CharT> fmt,
                                                     [[maybe_unused]] Args&&... args) {
  TEST_VALIDATE_EXCEPTION(
      std::format_error,
      [&]([[maybe_unused]] const std::format_error& e) {
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED(
                "\nFormat string   ", fmt, "\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      [&] {
        std::basic_string<CharT> out;
        std::vformat_to(std::back_inserter(out), std::locale(), fmt, std::make_format_args<context_t<CharT>>(args...));
      }());
};

int main(int, char**) {
  format_tests<char, execution_modus::partial>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_char_to_wchar_t(test);
  format_tests<wchar_t, execution_modus::partial>(test, test_exception);
#endif

  return 0;
}
