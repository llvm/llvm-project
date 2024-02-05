//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO PRINT Investigate see https://reviews.llvm.org/D156585
// UNSUPPORTED: no-filesystem

// XFAIL: availability-fp_to_chars-missing

// <ostream>

// template<class... Args>
//   void println(ostream& os, format_string<Args...> fmt, Args&&... args);

// [ostream.formatted.print]/3
//   If the function is vprint_unicode and os is a stream that refers to
//   a terminal capable of displaying Unicode which is determined in an
//   implementation-defined manner, writes out to the terminal using the
//   native Unicode API;
// This is tested in
// test/libcxx/input.output/iostream.format/output.streams/ostream.formatted/ostream.formatted.print/vprint_unicode.pass.cpp

#include <cassert>
#include <ostream>
#include <sstream>

#include "assert_macros.h"
#include "concat_macros.h"
#include "print_tests.h"
#include "test_format_string.h"
#include "test_macros.h"

auto test_file = []<class... Args>(std::string_view e, test_format_string<char, Args...> fmt, Args&&... args) {
  std::string expected = std::string{e} + '\n';

  std::stringstream sstr;
  std::println(sstr, fmt, std::forward<Args>(args)...);

  std::string out = sstr.str();
  TEST_REQUIRE(out == expected,
               TEST_WRITE_CONCATENATED(
                   "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
};

auto test_exception = []<class... Args>(std::string_view, std::string_view, Args&&...) {
  // After P2216 most exceptions thrown by std::format become ill-formed.
  // Therefore this tests does nothing.
  // A basic ill-formed test is done in format.verify.cpp
  // The exceptions are tested by other functions that don't use the basic-format-string as fmt argument.
};

int main(int, char**) {
  print_tests(test_file, test_exception);

  return 0;
}
