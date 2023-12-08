//===-- Unittests for the basic scanf converters --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/stdio/scanf_core/converter.h"
#include "src/stdio/scanf_core/core_structs.h"
#include "src/stdio/scanf_core/reader.h"
#include "src/stdio/scanf_core/string_reader.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcScanfConverterTest, RawMatchBasic) {
  const char *str = "abcdef";
  LIBC_NAMESPACE::scanf_core::StringReader str_reader(str);
  LIBC_NAMESPACE::scanf_core::Reader reader(&str_reader);

  // Reading "abc" should succeed.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "abc"),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(3));

  // Reading nothing should succeed and not advance.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, ""),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(3));

  // Reading a space where there is none should succeed and not advance.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, " "),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(3));

  // Reading "d" should succeed and advance by 1.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "d"),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(4));

  // Reading "z" should fail and not advance.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "z"),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::MATCHING_FAILURE));
  ASSERT_EQ(reader.chars_read(), size_t(4));

  // Reading "efgh" should fail but advance to the end.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "efgh"),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::MATCHING_FAILURE));
  ASSERT_EQ(reader.chars_read(), size_t(6));
}

TEST(LlvmLibcScanfConverterTest, RawMatchSpaces) {
  const char *str = " a \t\n b   cd";
  LIBC_NAMESPACE::scanf_core::StringReader str_reader(str);
  LIBC_NAMESPACE::scanf_core::Reader reader(&str_reader);

  // Reading "a" should fail and not advance.
  // Since there's nothing in the format string (the second argument to
  // raw_match) to match the space in the buffer it isn't consumed.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "a"),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::MATCHING_FAILURE));
  ASSERT_EQ(reader.chars_read(), size_t(0));

  // Reading "  \t\n  " should succeed and advance past the space.
  // Any number of space characters in the format string match 0 or more space
  // characters in the buffer.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "  \t\n  "),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(1));

  // Reading "ab" should fail and only advance past the a
  // The a characters match, but the format string doesn't have anything to
  // consume the spaces in the buffer, so it fails.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "ab"),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::MATCHING_FAILURE));
  ASSERT_EQ(reader.chars_read(), size_t(2));

  // Reading "  b" should succeed and advance past the b
  // Any number of space characters in the format string matches 0 or more space
  // characters in the buffer.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "  b"),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(7));

  // Reading "\t" should succeed and advance past the spaces to the c
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "\t"),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(10));

  // Reading "c d" should succeed and advance past the d.
  // Here the space character in the format string is matching 0 space
  // characters in the buffer.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::raw_match(&reader, "c d"),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(12));
}

TEST(LlvmLibcScanfConverterTest, StringConvSimple) {
  const char *str = "abcDEF123 654LKJihg";
  char result[20];
  LIBC_NAMESPACE::scanf_core::StringReader str_reader(str);
  LIBC_NAMESPACE::scanf_core::Reader reader(&str_reader);

  LIBC_NAMESPACE::scanf_core::FormatSection conv;
  conv.has_conv = true;
  conv.conv_name = 's';
  conv.output_ptr = result;

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(9));
  ASSERT_STREQ(result, "abcDEF123");

  //%s skips all spaces before beginning to read.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(19));
  ASSERT_STREQ(result, "654LKJihg");
}

TEST(LlvmLibcScanfConverterTest, StringConvNoWrite) {
  const char *str = "abcDEF123 654LKJihg";
  LIBC_NAMESPACE::scanf_core::StringReader str_reader(str);
  LIBC_NAMESPACE::scanf_core::Reader reader(&str_reader);

  LIBC_NAMESPACE::scanf_core::FormatSection conv;
  conv.has_conv = true;
  conv.conv_name = 's';
  conv.flags = LIBC_NAMESPACE::scanf_core::NO_WRITE;

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(9));

  //%s skips all spaces before beginning to read.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(19));
}

TEST(LlvmLibcScanfConverterTest, StringConvWidth) {
  const char *str = "abcDEF123 654LKJihg";
  char result[6];
  LIBC_NAMESPACE::scanf_core::StringReader str_reader(str);
  LIBC_NAMESPACE::scanf_core::Reader reader(&str_reader);

  LIBC_NAMESPACE::scanf_core::FormatSection conv;
  conv.has_conv = true;
  conv.conv_name = 's';
  conv.max_width = 5; // this means the result takes up 6 characters (with \0).
  conv.output_ptr = result;

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(5));
  ASSERT_STREQ(result, "abcDE");

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(9));
  ASSERT_STREQ(result, "F123");

  //%s skips all spaces before beginning to read.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(15));
  ASSERT_STREQ(result, "654LK");

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(19));
  ASSERT_STREQ(result, "Jihg");
}

TEST(LlvmLibcScanfConverterTest, CharsConv) {
  const char *str = "abcDEF123 654LKJihg MNOpqr&*(";
  char result[20];
  LIBC_NAMESPACE::scanf_core::StringReader str_reader(str);
  LIBC_NAMESPACE::scanf_core::Reader reader(&str_reader);

  LIBC_NAMESPACE::scanf_core::FormatSection conv;
  conv.has_conv = true;
  conv.conv_name = 'c';
  conv.output_ptr = result;

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(1));
  ASSERT_EQ(result[0], 'a');

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(2));
  ASSERT_EQ(result[0], 'b');

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(3));
  ASSERT_EQ(result[0], 'c');

  // Switch from character by character to 8 at a time.
  conv.max_width = 8;
  LIBC_NAMESPACE::cpp::string_view result_view(result, 8);

  //%c doesn't stop on spaces.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(11));
  ASSERT_EQ(result_view, LIBC_NAMESPACE::cpp::string_view("DEF123 6", 8));

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(19));
  ASSERT_EQ(result_view, LIBC_NAMESPACE::cpp::string_view("54LKJihg", 8));

  //%c also doesn't skip spaces at the start.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(27));
  ASSERT_EQ(result_view, LIBC_NAMESPACE::cpp::string_view(" MNOpqr&", 8));

  //%c will stop on a null byte though.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(29));
  ASSERT_EQ(LIBC_NAMESPACE::cpp::string_view(result, 2),
            LIBC_NAMESPACE::cpp::string_view("*(", 2));
}

TEST(LlvmLibcScanfConverterTest, ScansetConv) {
  const char *str = "abcDEF[123] 654LKJihg";
  char result[20];
  LIBC_NAMESPACE::scanf_core::StringReader str_reader(str);
  LIBC_NAMESPACE::scanf_core::Reader reader(&str_reader);

  LIBC_NAMESPACE::scanf_core::FormatSection conv;
  conv.has_conv = true;
  conv.conv_name = '[';
  conv.output_ptr = result;

  LIBC_NAMESPACE::cpp::bitset<256> bitset1;
  bitset1.set_range('a', 'c');
  bitset1.set_range('D', 'F');
  bitset1.set_range('1', '6');
  bitset1.set('[');
  bitset1.set(']');

  conv.scan_set = bitset1;

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(11));
  ASSERT_EQ(LIBC_NAMESPACE::cpp::string_view(result, 11),
            LIBC_NAMESPACE::cpp::string_view("abcDEF[123]", 11));

  // The scanset conversion doesn't consume leading spaces. If it did it would
  // return "654" here.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::MATCHING_FAILURE));
  ASSERT_EQ(reader.chars_read(), size_t(11));

  // This set is everything except for a-g.
  LIBC_NAMESPACE::cpp::bitset<256> bitset2;
  bitset2.set_range('a', 'g');
  bitset2.flip();
  conv.scan_set = bitset2;

  conv.max_width = 5;

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(16));
  ASSERT_EQ(LIBC_NAMESPACE::cpp::string_view(result, 5),
            LIBC_NAMESPACE::cpp::string_view(" 654L", 5));

  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(20));
  ASSERT_EQ(LIBC_NAMESPACE::cpp::string_view(result, 4),
            LIBC_NAMESPACE::cpp::string_view("KJih", 4));

  // This set is g and '\0'.
  LIBC_NAMESPACE::cpp::bitset<256> bitset3;
  bitset3.set('g');
  bitset3.set('\0');
  conv.scan_set = bitset3;

  // Even though '\0' is in the scanset, it should still stop on it.
  ASSERT_EQ(LIBC_NAMESPACE::scanf_core::convert(&reader, conv),
            static_cast<int>(LIBC_NAMESPACE::scanf_core::READ_OK));
  ASSERT_EQ(reader.chars_read(), size_t(21));
  ASSERT_EQ(LIBC_NAMESPACE::cpp::string_view(result, 1),
            LIBC_NAMESPACE::cpp::string_view("g", 1));
}
