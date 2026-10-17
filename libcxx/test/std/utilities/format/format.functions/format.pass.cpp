//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=9000000

// XFAIL: availability-fp_to_chars-missing

// Note this formatter shows additional information when tests are failing.
// This aids the development. Since other formatters fail in the same fashion
// they don't have this additional output.

// <format>

// template<class... Args>
//   string format(format-string<Args...> fmt, const Args&... args);
// template<class... Args>
//   wstring format(wformat-string<Args...> fmt, const Args&... args);

#include <format>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"
#include "string_literal.h"
#include "test_format_string.h"
#include "assert_macros.h"
#include "concat_macros.h"

auto test =
    []<class CharT, class... Args>(
        std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) constexpr {
      std::basic_string<CharT> out = std::format(fmt, std::forward<Args>(args)...);
      if (TEST_IS_CONSTANT_EVALUATED)
        assert(out == expected);
      else
        TEST_REQUIRE(
            out == expected,
            TEST_WRITE_CONCATENATED(
                "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
    };

auto test_exception = []<class CharT, class... Args>(std::string_view, std::basic_string_view<CharT>, Args&&...) {
  // After P2216 most exceptions thrown by std::format become ill-formed.
  // Therefore this tests does nothing.
  // A basic ill-formed test is done in format.verify.cpp
  // The exceptions are tested by other functions that don't use the basic-format-string as fmt argument.
};

#if TEST_STD_VER >= 26
constexpr bool test_constexpr() {
  format_tests_constexpr<char>(test, test_exception);
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_constexpr<wchar_t>(test, test_exception);
#  endif
  return true;
}
#endif

int main(int, char**) {
  format_tests<char, execution_modus::full>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_char_to_wchar_t(test);
  format_tests<wchar_t, execution_modus::full>(test, test_exception);
#endif

#if TEST_STD_VER >= 26
  static_assert(test_constexpr());
#endif

  return 0;
}
