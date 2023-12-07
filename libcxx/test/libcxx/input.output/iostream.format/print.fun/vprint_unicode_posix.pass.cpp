//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// REQUIRES: has-unix-headers

// <print>

// Tests the implementation of
//   void __print::__vprint_unicode_posix(FILE* __stream, string_view __fmt,
//                                        format_args __args, bool __write_nl,
//                                        bool __is_terminal);
//
// In the library when the stdout is redirected to a file it is no
// longer considered a terminal and the special terminal handling is no
// longer executed. By testing this function we can "force" emulate a
// terminal.
// Note __write_nl is tested by the public API.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <print>

#include "test_macros.h"

int main(int, char**) {
  std::array<char, 100> buffer;
  std::ranges::fill(buffer, '*');

  FILE* file = fmemopen(buffer.data(), buffer.size(), "wb");
  assert(file);

  // Test the file is buffered.
  std::fprintf(file, "Hello");
  assert(std::ftell(file) == 5);
#if defined(TEST_HAS_GLIBC) &&                                                                                         \
    !(__has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer))
  assert(std::ranges::all_of(buffer, [](char c) { return c == '*'; }));
#endif

  // Test writing to a "non-terminal" stream does not flush.
  std::__print::__vprint_unicode_posix(file, " world", std::make_format_args(), false, false);
  assert(std::ftell(file) == 11);
#if defined(TEST_HAS_GLIBC) &&                                                                                         \
    !(__has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer))
  assert(std::ranges::all_of(buffer, [](char c) { return c == '*'; }));
#endif

  // Test writing to a "terminal" stream flushes before writing.
  std::__print::__vprint_unicode_posix(file, "!", std::make_format_args(), false, true);
  assert(std::ftell(file) == 12);
  assert(std::string_view(buffer.data(), buffer.data() + 11) == "Hello world");
#if defined(TEST_HAS_GLIBC)
  // glibc does not flush after a write.
  assert(buffer[11] != '!');
#endif

  // Test everything is written when closing the stream.
  std::fclose(file);
  assert(std::string_view(buffer.data(), buffer.data() + 12) == "Hello world!");

  return 0;
}
