//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// The error exception has no system error string.
// XFAIL: LIBCXX-ANDROID-FIXME

// <print>

// template<class... Args>
//   void print(FILE* stream, format_string<Args...> fmt, Args&&... args);

// In the library when the stdout is redirected to a file it is no
// longer considered a terminal and the special terminal handling is no
// longer executed. There are tests in
//   libcxx/test/libcxx/input.output/iostream.format/print.fun/
// to validate that behaviour

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <print>
#include <string_view>

#include "assert_macros.h"
#include "concat_macros.h"
#include "filesystem_test_helper.h"
#include "print_tests.h"
#include "test_format_string.h"
#include "test_macros.h"

scoped_test_env env;
std::string filename = env.create_file("output.txt");

auto test_file = []<class... Args>(std::string_view expected, test_format_string<char, Args...> fmt, Args&&... args) {
  FILE* file = fopen(filename.c_str(), "wb");
  assert(file);

  std::print(file, fmt, std::forward<Args>(args)...);
  std::fclose(file);

  std::ifstream stream{filename.c_str(), std::ios_base::in | std::ios_base::binary};
  std::string out(std::istreambuf_iterator<char>{stream}, {});
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

// Glibc fails writing to a wide stream.
#if defined(TEST_HAS_GLIBC) && !defined(TEST_HAS_NO_WIDE_CHARACTERS)
static void test_wide_stream() {
  FILE* file = fopen(filename.c_str(), "wb");
  assert(file);

  int mode = std::fwide(file, 1);
  assert(mode > 0);

  TEST_VALIDATE_EXCEPTION(
      std::system_error,
      [&]([[maybe_unused]] const std::system_error& e) {
        [[maybe_unused]] std::string_view what{"failed to write formatted output"};
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      std::print(file, "hello"));
}
#endif // defined(TEST_HAS_GLIBC) && !defined(TEST_HAS_NO_WIDE_CHARACTERS)

static void test_read_only() {
  FILE* file = fopen(filename.c_str(), "r");
  assert(file);

  TEST_VALIDATE_EXCEPTION(
      std::system_error,
      [&]([[maybe_unused]] const std::system_error& e) {
        [[maybe_unused]] std::string_view what{
            "failed to write formatted output: " TEST_IF_AIX("Broken pipe", "Operation not permitted")};
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      std::print(file, "hello"));
}

static void test_new_line() {
  // Text does newline translation.
  {
    FILE* file = fopen(filename.c_str(), "w");
    assert(file);

    std::print(file, "\n");
#ifndef _WIN32
    assert(std::ftell(file) == 1);
#else
    assert(std::ftell(file) == 2);
#endif
  }
  // Binary no newline translation.
  {
    FILE* file = fopen(filename.c_str(), "wb");
    assert(file);

    std::print(file, "\n");
    assert(std::ftell(file) == 1);
  }
}

int main(int, char**) {
  print_tests(test_file, test_exception);

#if defined(TEST_HAS_GLIBC) && !defined(TEST_HAS_NO_WIDE_CHARACTERS)
  test_wide_stream();
#endif
  test_read_only();
  test_new_line();

  return 0;
}
