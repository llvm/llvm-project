//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// Basic test to validate ill-formed code is properly detected.

// <format>

// template<class... Args>
//   size_t formatted_size(const locale& loc,
//                         format-string<Args...> fmt, const Args&... args);
// template<class... Args>
//   size_t formatted_size(const locale& loc,
//                         wformat-string<Args...> fmt, const Args&... args);

#include <format>
#include <locale>

#include "test_macros.h"

// clang-format off

void f() {
  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "{"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "{}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "{0}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "{:-}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "{:#}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "{:L}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "{0:{0}}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "{:.42d}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), "{:d}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"{"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"{}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"{0}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"{:-}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"{:#}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"{:L}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"{0:{0}}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"{:.42d}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::formatted_size(std::locale(), L"{:d}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}
#endif
}
