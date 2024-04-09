//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: executor-has-no-bash
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// FIXME PRINT How to test println on Windows?
// XFAIL: msvc, target={{.+}}-windows-gnu

// XFAIL: availability-fp_to_chars-missing

// <print>

//   void println();

// Testing this properly is quite hard; the function unconditionally
// writes to stdout. When stdout is redirected to a file it is no longer
// considered a terminal. The function is a small wrapper around
//
// template<class... Args>
//   void println(FILE* stream, format_string<Args...> fmt, Args&&... args);
//
// So do minimal tests for this function and rely on the FILE* overload
// to do more testing.
//
// The testing is based on the testing for std::cout.

// TODO PRINT Use lit builtin echo

// FILE_DEPENDENCIES: echo.sh
// RUN: %{build}
// RUN: %{exec} bash echo.sh -ne "println blank line test: \n" > %t.expected
// RUN: %{exec} "%t.exe" > %t.actual
// RUN: diff -u %t.actual %t.expected

#include <print>

int main(int, char**) {
  // On some configurations the `diff -u` test fails if we print a single blank line character `\n`, so we print some text first.
  std::print("println blank line test: ");
  std::println();

  return 0;
}
