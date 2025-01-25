//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: msvc, target={{.+}}-windows-gnu
// XFAIL: availability-fp_to_chars-missing

// fmemopen is available starting in Android M (API 23)
// XFAIL: target={{.+}}-android{{(eabi)?(21|22)}}

// <print>

// The FILE returned by fmemopen does not have file descriptor.
// This means the test could fail when the implementation uses a
// function that requires a file descriptor, for example write.
//
// This tests all print functions which takes a FILE* as argument.

// template<class... Args>
//   void print(FILE* stream, format_string<Args...> fmt, Args&&... args);
// void println();                                                          // Since C++26
// template<class... Args>
//   void println(FILE* stream, format_string<Args...> fmt, Args&&... args);
// void println(FILE* stream);                                              // Since C++26
// void vprint_unicode(FILE* stream, string_view fmt, format_args args);
// void vprint_nonunicode(FILE* stream, string_view fmt, format_args args);

#include <array>
#include <cstdio>
#include <cassert>
#include <print>

static void test_print() {
  std::array<char, 100> buffer{0};

  FILE* file = fmemopen(buffer.data(), buffer.size(), "wb");
  assert(file);

  std::print(file, "hello world{}", '!');
  long pos = std::ftell(file);
  std::fclose(file);

  assert(pos > 0);
  assert(std::string_view(buffer.data(), pos) == "hello world!");
}

static void test_println() {
  std::array<char, 100> buffer{0};

  FILE* file = fmemopen(buffer.data(), buffer.size(), "wb");
  assert(file);

  std::println(file, "hello world{}", '!');
  long pos = std::ftell(file);
  std::fclose(file);

  assert(pos > 0);
  assert(std::string_view(buffer.data(), pos) == "hello world!\n");
}

static void test_println_blank_line() {
  std::array<char, 100> buffer{0};

  FILE* file = fmemopen(buffer.data(), buffer.size(), "wb");
  assert(file);

  std::println(file);
  long pos = std::ftell(file);
  std::fclose(file);

  assert(pos > 0);
  assert(std::string_view(buffer.data(), pos) == "\n");
}

static void test_vprint_unicode() {
#ifdef TEST_HAS_NO_UNICODE
  std::array<char, 100> buffer{0};

  FILE* file = fmemopen(buffer.data(), buffer.size(), "wb");
  assert(file);

  char c = '!';
  std::vprint_unicode(file, "hello world{}", std::make_format_args(c));
  long pos = std::ftell(file);
  std::fclose(file);

  assert(pos > 0);
  assert(std::string_view(buffer.data(), pos) == "hello world!");
#endif // TEST_HAS_NO_UNICODE
}

static void test_vprint_nonunicode() {
  std::array<char, 100> buffer{0};

  FILE* file = fmemopen(buffer.data(), buffer.size(), "wb");
  assert(file);

  char c = '!';
  std::vprint_nonunicode(file, "hello world{}", std::make_format_args(c));
  long pos = std::ftell(file);
  std::fclose(file);

  assert(pos > 0);
  assert(std::string_view(buffer.data(), pos) == "hello world!");
}

int main(int, char**) {
  test_print();
  test_println();
  test_println_blank_line();
  test_vprint_unicode();
  test_vprint_nonunicode();

  return 0;
}
