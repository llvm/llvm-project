//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-format

// This test requires the dylib support introduced in D92214.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{.+}}
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx11.{{.+}}

// <format>

// template<class charT, formattable<charT>... Ts>
//   struct formatter<pair-or-tuple<Ts...>, charT>
//
// tested in the format functions
//
// string vformat(string_view fmt, format_args args);
// wstring vformat(wstring_view fmt, wformat_args args);

#include <format>
#include <cassert>

#include "test_macros.h"
#include "format.functions.tests.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <iostream>
#  include <concepts>
#endif

auto test = []<class CharT, class... Args>(
                std::basic_string_view<CharT> expected, std::basic_string_view<CharT> fmt, Args&&... args) {
  std::basic_string<CharT> out = std::vformat(fmt, std::make_format_args<context_t<CharT>>(args...));
#ifndef TEST_HAS_NO_LOCALIZATION
  if constexpr (std::same_as<CharT, char>)
    if (out != expected)
      std::cerr << "\nFormat string   " << fmt << "\nExpected output " << expected << "\nActual output   " << out
                << '\n';
#endif // TEST_HAS_NO_LOCALIZATION
  assert(out == expected);
};

auto test_exception =
    []<class CharT, class... Args>(
        [[maybe_unused]] std::string_view what,
        [[maybe_unused]] std::basic_string_view<CharT> fmt,
        [[maybe_unused]] Args&&... args) {
#ifndef TEST_HAS_NO_EXCEPTIONS
      try {
        TEST_IGNORE_NODISCARD std::vformat(fmt, std::make_format_args<context_t<CharT>>(args...));
#  if !defined(TEST_HAS_NO_LOCALIZATION)
        if constexpr (std::same_as<CharT, char>)
          std::cerr << "\nFormat string   " << fmt << "\nDidn't throw an exception.\n";
#  endif //  !defined(TEST_HAS_NO_LOCALIZATION
        assert(false);
      } catch ([[maybe_unused]] const std::format_error& e) {
#  if defined(_LIBCPP_VERSION)
#    if !defined(TEST_HAS_NO_LOCALIZATION)
        if constexpr (std::same_as<CharT, char>) {
          if (e.what() != what)
            std::cerr << "\nFormat string   " << fmt << "\nExpected exception " << what << "\nActual exception   "
                      << e.what() << '\n';
        }
#    endif // !defined(TEST_HAS_NO_LOCALIZATION
        assert(e.what() == what);
#  endif   // defined(_LIBCPP_VERSION)
        return;
      }
      assert(false);
#endif
    };

int main(int, char**) {
  run_tests<char>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  run_tests<wchar_t>(test, test_exception);
#endif

  return 0;
}
