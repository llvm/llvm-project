//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// XFAIL: availability-fp_to_chars-missing

// <format>

// [format.string.std]/8
// If { arg-idopt } is used in a width or precision, the value of the
// corresponding formatting argument is used in its place. If the
// corresponding formatting argument is not of standard signed or unsigned
// integer type, or its value is negative for precision or non-positive for
// width, an exception of type format_error is thrown.
//
// This test does the compile-time validation

#include <format>

#include "test_macros.h"

void test_char_width() {
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:{}}", 42, true);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:{}}", 42, '0');
#ifndef TEST_HAS_NO_INT128
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:{}}", 42, __int128_t(0));
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:{}}", 42, __uint128_t(0));
#endif
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:{}}", 42, 42.0f);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:{}}", 42, 42.0);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:{}}", 42, 42.0l);
}

void test_char_precision() {
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:0.{}}", 42.0, true);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:0.{}}", 42.0, '0');
#ifndef TEST_HAS_NO_INT128
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:0.{}}", 42.0, __int128_t(0));
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:0.{}}", 42.0, __uint128_t(0));
#endif
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:0.{}}", 42.0, 42.0f);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:0.{}}", 42.0, 42.0);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format("{:0.{}}", 42.0, 42.0l);
}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
void test_wchar_t_width() {
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:{}}", 42, true);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:{}}", 42, '0');
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:{}}", 42, L'0');
#  ifndef TEST_HAS_NO_INT128
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:{}}", 42, __int128_t(0));
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:{}}", 42, __uint128_t(0));
#  endif
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:{}}", 42, 42.0f);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:{}}", 42, 42.0);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:{}}", 42, 42.0l);
}

void test_wchar_t_precision() {
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:0.{}}", 42.0, true);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:0.{}}", 42.0, '0');
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:0.{}}", 42.0, L'0');
#  ifndef TEST_HAS_NO_INT128
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:0.{}}", 42.0, __int128_t(0));
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:0.{}}", 42.0, __uint128_t(0));
#  endif
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:0.{}}", 42.0, 42.0f);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:0.{}}", 42.0, 42.0);
  // expected-error-re@*:* {{call to consteval function {{.*}} is not a constant expression}}
  TEST_IGNORE_NODISCARD std::format(L"{:0.{}}", 42.0, 42.0l);
}

#endif // TEST_HAS_NO_WIDE_CHARACTERS
