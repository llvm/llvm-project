//===----------------------------------------------------------------------===//
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

// template<class Out, class... Args>
//   format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                       const locale& loc, format-string<Args...> fmt,
//                                       const Args&... args);
// template<class Out, class... Args>
//   format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                       const locale& loc, wformat-string<Args...> fmt,
//                                       const Args&... args);

#include <format>
#include <algorithm>
#include <cassert>
#include <list>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"
#include "string_literal.h"
#include "test_format_string.h"

auto test =
    []<class CharT, class... Args>(
        std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) constexpr {
      {
        std::list<CharT> out;
        std::format_to_n_result result =
            std::format_to_n(std::back_inserter(out), 0, std::locale(), fmt, std::forward<Args>(args)...);
        // To avoid signedness warnings make sure formatted_size uses the same type
        // as result.size.
        using diff_type          = decltype(result.size);
        diff_type formatted_size = std::formatted_size(std::locale(), fmt, std::forward<Args>(args)...);

        assert(result.size == formatted_size);
        assert(out.empty());
      }
      {
        std::vector<CharT> out;
        std::format_to_n_result result =
            std::format_to_n(std::back_inserter(out), 5, std::locale(), fmt, std::forward<Args>(args)...);
        using diff_type          = decltype(result.size);
        diff_type formatted_size = std::formatted_size(std::locale(), fmt, std::forward<Args>(args)...);
        diff_type size           = std::min<diff_type>(5, formatted_size);

        assert(result.size == formatted_size);
        assert(std::equal(out.begin(), out.end(), expected.begin(), expected.begin() + size));
      }
      {
        std::basic_string<CharT> out;
        std::format_to_n_result result =
            std::format_to_n(std::back_inserter(out), 1000, std::locale(), fmt, std::forward<Args>(args)...);
        using diff_type          = decltype(result.size);
        diff_type formatted_size = std::formatted_size(std::locale(), fmt, std::forward<Args>(args)...);
        diff_type size           = std::min<diff_type>(1000, formatted_size);

        assert(result.size == formatted_size);
        assert(out == expected.substr(0, size));
      }
      {
        // Test the returned iterator.
        std::basic_string<CharT> out(10, CharT(' '));
        std::format_to_n_result result =
            std::format_to_n(out.begin(), 10, std::locale(), fmt, std::forward<Args>(args)...);
        using diff_type          = decltype(result.size);
        diff_type formatted_size = std::formatted_size(std::locale(), fmt, std::forward<Args>(args)...);
        diff_type size           = std::min<diff_type>(10, formatted_size);

        assert(result.size == formatted_size);
        assert(result.out == out.begin() + size);
        assert(out.substr(0, size) == expected.substr(0, size));
      }
      {
        static_assert(std::is_signed_v<std::iter_difference_t<CharT*>>,
                      "If the difference type isn't negative the test will fail "
                      "due to using a large positive value.");
        CharT buffer[1]                = {CharT(0)};
        std::format_to_n_result result = std::format_to_n(buffer, -1, std::locale(), fmt, std::forward<Args>(args)...);
        using diff_type                = decltype(result.size);
        diff_type formatted_size       = std::formatted_size(std::locale(), fmt, std::forward<Args>(args)...);

        assert(result.size == formatted_size);
        assert(result.out == buffer);
        assert(buffer[0] == CharT(0));
      }
    };

auto test_exception = []<class CharT, class... Args>(std::string_view, std::basic_string_view<CharT>, Args&&...) {
  // After P2216 most exceptions thrown by std::format_to_n become ill-formed.
  // Therefore this tests does nothing.
  // A basic ill-formed test is done in format_to_n.locale.verify.cpp
  // The exceptions are tested by other functions that don't use the basic-format-string as fmt argument.
};

int main(int, char**) {
  format_tests<char, execution_modus::partial>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_char_to_wchar_t(test);
  format_tests<wchar_t, execution_modus::partial>(test, test_exception);
#endif

  return 0;
}
