//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// Basic test to validate ill-formed code is properly detected.

// <format>

// template<class Out, class... Args>
//   format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                       format-string<Args...> fmt, const Args&... args);
// template<class Out, class... Args>
//   format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                       wformat-string<Args...> fmt, const Args&... args);

#include <format>

#include "test_macros.h"

extern char* out;
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
extern wchar_t* wout;
#endif

// clang-format off

void f() {
  std::format_to_n(out, 42, "{"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(out, 42, "}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(out, 42, "{}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(out, 42, "{0}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(out, 42, "{:-}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(out, 42, "{:#}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(out, 42, "{:L}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(out, 42, "{0:{0}}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(out, 42, "{:.42d}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(out, 42, "{:d}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::format_to_n(wout, 42, L"{"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(wout, 42, L"}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(wout, 42, L"{}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(wout, 42, L"{0}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(wout, 42, L"{:-}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(wout, 42, L"{:#}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(wout, 42, L"{:L}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(wout, 42, L"{0:{0}}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(wout, 42, L"{:.42d}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to_n(wout, 42, L"{:d}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}
#endif
}
