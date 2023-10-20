//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO PRINT Enable again
// https://reviews.llvm.org/D150044
// https://lab.llvm.org/buildbot/#/builders/237/builds/3578
// UNSUPPORTED: asan, hwasan, msan

// XFAIL: availability-fp_to_chars-missing

// The error exception has no system error string.
// XFAIL: LIBCXX-ANDROID-FIXME

// <print>

// void vprint_nonunicode(FILE* stream, string_view fmt, format_args args);

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
#include "test_macros.h"

scoped_test_env env;
std::string filename = env.create_file("output.txt");

auto test_file = []<class... Args>(std::string_view expected, std::string_view fmt, Args&&... args) {
  FILE* file = fopen(filename.c_str(), "wb");
  assert(file);

  std::vprint_nonunicode(file, fmt, std::make_format_args(args...));
  std::fclose(file);

  std::ifstream stream{filename.c_str(), std::ios_base::in | std::ios_base::binary};
  std::string out(std::istreambuf_iterator<char>{stream}, {});
  TEST_REQUIRE(out == expected,
               TEST_WRITE_CONCATENATED(
                   "\nFormat string   ", fmt, "\nExpected output ", expected, "\nActual output   ", out, '\n'));
};

auto test_exception = []<class... Args>([[maybe_unused]] std::string_view what,
                                        [[maybe_unused]] std::string_view fmt,
                                        [[maybe_unused]] Args&&... args) {
  FILE* file = fopen(filename.c_str(), "wb");
  assert(file);

  TEST_VALIDATE_EXCEPTION(
      std::format_error,
      [&]([[maybe_unused]] const std::format_error& e) {
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED(
                "\nFormat string   ", fmt, "\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      std::vprint_nonunicode(file, fmt, std::make_format_args(args...)));

  fclose(file);
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
      std::vprint_nonunicode(file, "hello", std::make_format_args()));
}
#endif // defined(TEST_HAS_GLIBC) && !defined(TEST_HAS_NO_WIDE_CHARACTERS)

static void test_read_only() {
  FILE* file = fopen(filename.c_str(), "r");
  assert(file);

  TEST_VALIDATE_EXCEPTION(
      std::system_error,
      [&]([[maybe_unused]] const std::system_error& e) {
#ifdef _AIX
        [[maybe_unused]] std::string_view what{"failed to write formatted output: Broken pipe"};
#else
        [[maybe_unused]] std::string_view what{"failed to write formatted output: Operation not permitted"};
#endif
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      std::vprint_nonunicode(file, "hello", std::make_format_args()));
}

static void test_new_line() {
  // Text does newline translation.
  {
    FILE* file = fopen(filename.c_str(), "w");
    assert(file);

    std::vprint_nonunicode(file, "\n", std::make_format_args());
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

    std::vprint_nonunicode(file, "\n", std::make_format_args());
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
