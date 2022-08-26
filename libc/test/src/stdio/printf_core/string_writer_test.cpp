//===-- Unittests for the printf String Writer ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/stdio/printf_core/string_writer.h"
#include "src/stdio/printf_core/writer.h"

#include "utils/UnitTest/Test.h"

using __llvm_libc::cpp::string_view;

__llvm_libc::printf_core::Writer get_writer(void *str_writer) {
  return __llvm_libc::printf_core::Writer(
      str_writer, __llvm_libc::printf_core::StringWriter::write_str,
      __llvm_libc::printf_core::StringWriter::write_chars,
      __llvm_libc::printf_core::StringWriter::write_char);
}

TEST(LlvmLibcPrintfStringWriterTest, Constructor) {
  char str[10];
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
}

TEST(LlvmLibcPrintfStringWriterTest, Write) {
  char str[4] = {'D', 'E', 'F', 'G'};
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  writer.write({"abc", 3});

  EXPECT_EQ(str[3], 'G');
  // This null terminates the string. The writer has no indication when the
  // string is done, so it relies on the user to tell it when to null terminate
  // the string. Importantly, it can't tell the difference between an intended
  // max length of 0 (write nothing) or 1 (write just a null byte), and so it
  // relies on the caller to do that bounds check.
  str_writer.terminate();

  ASSERT_STREQ("abc", str);
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST(LlvmLibcPrintfStringWriterTest, WriteMultipleTimes) {
  char str[10];
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  writer.write({"abc", 3});
  writer.write({"DEF", 3});
  writer.write({"1234", 3});

  str_writer.terminate();

  ASSERT_STREQ("abcDEF123", str);
  ASSERT_EQ(writer.get_chars_written(), 9);
}

TEST(LlvmLibcPrintfStringWriterTest, WriteChars) {
  char str[4] = {'D', 'E', 'F', 'G'};
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  writer.write('a', 3);

  EXPECT_EQ(str[3], 'G');
  str_writer.terminate();

  ASSERT_STREQ("aaa", str);
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST(LlvmLibcPrintfStringWriterTest, WriteCharsMultipleTimes) {
  char str[10];
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  writer.write('a', 3);
  writer.write('D', 3);
  writer.write('1', 3);

  str_writer.terminate();

  ASSERT_STREQ("aaaDDD111", str);
  ASSERT_EQ(writer.get_chars_written(), 9);
}

TEST(LlvmLibcPrintfStringWriterTest, WriteManyChars) {
  char str[100];
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  writer.write('Z', 99);

  str_writer.terminate();

  ASSERT_STREQ("ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZZ"
               "ZZZZZZZZZ",
               str);
  ASSERT_EQ(writer.get_chars_written(), 99);
}

TEST(LlvmLibcPrintfStringWriterTest, MixedWrites) {
  char str[13];
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  str_writer.terminate();

  ASSERT_STREQ("aaaDEF111456", str);
  ASSERT_EQ(writer.get_chars_written(), 12);
}

TEST(LlvmLibcPrintfStringWriterTest, WriteWithMaxLength) {
  char str[11];
  __llvm_libc::printf_core::StringWriter str_writer(str, 10);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  writer.write({"abcDEF123456", 12});

  str_writer.terminate();

  ASSERT_STREQ("abcDEF1234", str);
  ASSERT_EQ(writer.get_chars_written(), 12);
}

TEST(LlvmLibcPrintfStringWriterTest, WriteCharsWithMaxLength) {
  char str[11];
  __llvm_libc::printf_core::StringWriter str_writer(str, 10);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));

  writer.write('1', 15);

  str_writer.terminate();

  ASSERT_STREQ("1111111111", str);
  ASSERT_EQ(writer.get_chars_written(), 15);
}

TEST(LlvmLibcPrintfStringWriterTest, MixedWriteWithMaxLength) {
  char str[11];
  __llvm_libc::printf_core::StringWriter str_writer(str, 10);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  str_writer.terminate();

  ASSERT_STREQ("aaaDEF1114", str);
  ASSERT_EQ(writer.get_chars_written(), 12);
}

TEST(LlvmLibcPrintfStringWriterTest, StringWithMaxLengthOne) {
  char str[1];
  __llvm_libc::printf_core::StringWriter str_writer(str, 0);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  // This is because the max length should be at most 1 less than the size of
  // the buffer it's writing to.
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  str_writer.terminate();

  ASSERT_STREQ("", str);
  ASSERT_EQ(writer.get_chars_written(), 12);
}

TEST(LlvmLibcPrintfStringWriterTest, NullStringWithZeroMaxLength) {
  __llvm_libc::printf_core::StringWriter str_writer(nullptr, 0);
  __llvm_libc::printf_core::Writer writer =
      get_writer(reinterpret_cast<void *>(&str_writer));
  writer.write('a', 3);
  writer.write({"DEF", 3});
  writer.write('1', 3);
  writer.write({"456", 3});

  ASSERT_EQ(writer.get_chars_written(), 12);
}
