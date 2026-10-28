//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// ile
/// Unittests for get_signal_string.
///
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/StringUtil/signal_to_string.h"
#include "src/__support/integer_to_string.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::span;
using LIBC_NAMESPACE::cpp::string_view;

TEST(LlvmLibcSignalToStringTest, BuiltDescription) {
  char buffer[64];
  for (char &c : buffer)
    c = 'x';

  string_view result =
      LIBC_NAMESPACE::get_signal_string(12345, span<char>(buffer, 64));
  ASSERT_TRUE(result == "Unknown signal 12345");

  result =
      LIBC_NAMESPACE::get_signal_string(SIGRTMIN + 1, span<char>(buffer, 64));
  ASSERT_TRUE(result == "Real-time signal 1");
}

TEST(LlvmLibcSignalToStringTest, BuiltCStringIsTerminated) {
  char buffer[64];
  for (char &c : buffer)
    c = 'x';
  ASSERT_STREQ(
      LIBC_NAMESPACE::get_signal_cstring(12345, span<char>(buffer, 64)),
      "Unknown signal 12345");

  for (char &c : buffer)
    c = 'x';
  ASSERT_STREQ(
      LIBC_NAMESPACE::get_signal_cstring(SIGRTMIN + 1, span<char>(buffer, 64)),
      "Real-time signal 1");
}

TEST(LlvmLibcSignalToStringTest, SmallBufferGivesTheBaseDescription) {
  char buffer[4];
  string_view result =
      LIBC_NAMESPACE::get_signal_string(12345, span<char>(buffer, 4));
  ASSERT_TRUE(result == "Unknown signal");
  ASSERT_STREQ(LIBC_NAMESPACE::get_signal_cstring(12345, span<char>(buffer, 4)),
               "Unknown signal");
}

TEST(LlvmLibcSignalToStringTest,
     NoRoomForTheTerminatorGivesTheBaseDescription) {
  // Room for the longest text but not for a terminator after it.
  constexpr size_t SIZE = sizeof("Unknown signal") +
                          LIBC_NAMESPACE::IntegerToString<int>::buffer_size();
  char buffer[SIZE + 1];
  for (char &c : buffer)
    c = 'x';

  string_view result =
      LIBC_NAMESPACE::get_signal_string(2147483647, span<char>(buffer, SIZE));
  ASSERT_TRUE(result == "Unknown signal");
  ASSERT_EQ(buffer[SIZE], 'x');
}
