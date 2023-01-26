//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// libc++ supports basic_format_string in C++20 as an extension
// UNSUPPORTED: !stdlib=libc++ && c++20

// <format>

// template<class charT, class... Args>
// class basic_format_string<charT, type_identity_t<Args>...>
//
// template<class T> consteval basic_format_string(const T& s);
//
// This constructor does the compile-time format string validation for the
// std::format* functions.

#include <format>

#include <string_view>

#include "test_macros.h"

void run() {
  (void)std::basic_format_string<char>{"foo"};
  (void)std::basic_format_string<char>{"{}"}; // expected-error-re {{call to consteval function{{.*}}is not a constant expression}}
  (void)std::basic_format_string<char, int>{"{0:{0}P}"}; // expected-error-re {{call to consteval function{{.*}}is not a constant expression}}
  (void)std::basic_format_string<char, int>{"{0:{0}}"};
  (void)std::basic_format_string<char, float>{"{0:{0}}"}; // expected-error-re {{call to consteval function{{.*}}is not a constant expression}}
  (void)std::basic_format_string<char, int>{"{.3}"}; // expected-error-re {{call to consteval function{{.*}}is not a constant expression}}
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  (void)std::basic_format_string<wchar_t>{L"foo"};
  (void)std::basic_format_string<wchar_t>{L"{}"}; // expected-error-re {{call to consteval function{{.*}}is not a constant expression}}
  (void)std::basic_format_string<wchar_t, int>{L"{0:{0}P}"}; // expected-error-re {{call to consteval function{{.*}}is not a constant expression}}
  (void)std::basic_format_string<wchar_t, int>{L"{0:{0}}"};
  (void)std::basic_format_string<wchar_t, float>{L"{0:{0}}"}; // expected-error-re {{call to consteval function{{.*}}is not a constant expression}}
  (void)std::basic_format_string<wchar_t, int>{L"{.3}"}; // expected-error-re {{call to consteval function{{.*}}is not a constant expression}}
#endif
}
