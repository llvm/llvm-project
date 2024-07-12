//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: libcpp-has-no-unicode
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// <print>

//  void vprint_unicode(string_view fmt, format_args args);

// Testing this properly is quite hard; the function unconditionally
// writes to stdout. When stdout is redirected to a file it is no longer
// considered a terminal. The function is a small wrapper around
//
//  void vprint_unicode(FILE* stream, string_view fmt, format_args args);
//
// So do minimal tests for this function and rely on the FILE* overload
// to do more testing.
//
// The testing is based on the testing for std::cout.

// RUN: %{build}
// RUN: echo -n "1234 一二三四 true 0x0" > %t.expected
// RUN: %{exec} %t.exe > %t.actual
// RUN: diff -u %t.actual %t.expected

#include <print>

int main(int, char**) {
  // The data is passed as-is so it does not depend on the encoding of the input.
  int i         = 1234;
  const char* s = "一二三四";
  bool b        = true;
  nullptr_t p   = nullptr;
  std::vprint_unicode("{} {} ", std::make_format_args(i, s));
  std::vprint_unicode("{} {}", std::make_format_args(b, p));

  return 0;
}
