//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

#include <format>

#include <utility>
#include <tuple>

#include "test_macros.h"

// clang-format off

void f() {
  TEST_IGNORE_NODISCARD std::format("{::}", std::make_tuple(0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::format("{::^}", std::make_tuple(0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::format("{:+}", std::make_pair(0, 0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::format("{:m}", std::make_tuple(0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::format("{:m}", std::make_tuple(0, 0, 0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  TEST_IGNORE_NODISCARD std::format(L"{::}", std::make_tuple(0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::format(L"{::^}", std::make_tuple(0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::format(L"{:+}", std::make_pair(0, 0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::format(L"{:m}", std::make_tuple(0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  TEST_IGNORE_NODISCARD std::format(L"{:m}", std::make_tuple(0, 0, 0)); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}
#endif
}
