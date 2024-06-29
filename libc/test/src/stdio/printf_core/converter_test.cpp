//===-- Unittests for the printf Converter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/converter.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

#include "test/UnitTest/Test.h"

class LlvmLibcPrintfConverterTest : public LIBC_NAMESPACE::testing::Test {
protected:
  // void SetUp() override {}
  // void TearDown() override {}

  char str[60];
  LIBC_NAMESPACE::printf_core::WriteBuffer wb =
      LIBC_NAMESPACE::printf_core::WriteBuffer(str, sizeof(str) - 1);
  LIBC_NAMESPACE::printf_core::Writer writer =
      LIBC_NAMESPACE::printf_core::Writer(&wb);
};

TEST_F(LlvmLibcPrintfConverterTest, SimpleRawConversion) {
  LIBC_NAMESPACE::printf_core::FormatSection raw_section;
  raw_section.has_conv = false;
  raw_section.raw_string = "abc";

  LIBC_NAMESPACE::printf_core::convert(&writer, raw_section);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "abc");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcPrintfConverterTest, PercentConversion) {
  LIBC_NAMESPACE::printf_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%%";
  simple_conv.conv_name = '%';

  LIBC_NAMESPACE::printf_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "%");
  ASSERT_EQ(writer.get_chars_written(), 1);
}

TEST_F(LlvmLibcPrintfConverterTest, CharConversionSimple) {
  LIBC_NAMESPACE::printf_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  // If has_conv is true, the raw string is ignored. They are not being parsed
  // and match the actual conversion taking place so that you can compare these
  // tests with other implmentations. The raw strings are completely optional.
  simple_conv.raw_string = "%c";
  simple_conv.conv_name = 'c';
  simple_conv.conv_val_raw = 'D';

  LIBC_NAMESPACE::printf_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "D");
  ASSERT_EQ(writer.get_chars_written(), 1);
}

TEST_F(LlvmLibcPrintfConverterTest, CharConversionRightJustified) {
  LIBC_NAMESPACE::printf_core::FormatSection right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = "%4c";
  right_justified_conv.conv_name = 'c';
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_raw = 'E';
  LIBC_NAMESPACE::printf_core::convert(&writer, right_justified_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "   E");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcPrintfConverterTest, CharConversionLeftJustified) {
  LIBC_NAMESPACE::printf_core::FormatSection left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = "%-4c";
  left_justified_conv.conv_name = 'c';
  left_justified_conv.flags =
      LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_raw = 'F';
  LIBC_NAMESPACE::printf_core::convert(&writer, left_justified_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "F   ");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionSimple) {

  LIBC_NAMESPACE::printf_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%s";
  simple_conv.conv_name = 's';
  simple_conv.conv_val_ptr = const_cast<char *>("DEF");

  LIBC_NAMESPACE::printf_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "DEF");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionPrecisionHigh) {
  LIBC_NAMESPACE::printf_core::FormatSection high_precision_conv;
  high_precision_conv.has_conv = true;
  high_precision_conv.raw_string = "%4s";
  high_precision_conv.conv_name = 's';
  high_precision_conv.precision = 4;
  high_precision_conv.conv_val_ptr = const_cast<char *>("456");
  LIBC_NAMESPACE::printf_core::convert(&writer, high_precision_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "456");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionPrecisionLow) {
  LIBC_NAMESPACE::printf_core::FormatSection low_precision_conv;
  low_precision_conv.has_conv = true;
  low_precision_conv.raw_string = "%.2s";
  low_precision_conv.conv_name = 's';
  low_precision_conv.precision = 2;
  low_precision_conv.conv_val_ptr = const_cast<char *>("xyz");
  LIBC_NAMESPACE::printf_core::convert(&writer, low_precision_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "xy");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionRightJustified) {
  LIBC_NAMESPACE::printf_core::FormatSection right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = "%4s";
  right_justified_conv.conv_name = 's';
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_ptr = const_cast<char *>("789");
  LIBC_NAMESPACE::printf_core::convert(&writer, right_justified_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, " 789");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionLeftJustified) {
  LIBC_NAMESPACE::printf_core::FormatSection left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = "%-4s";
  left_justified_conv.conv_name = 's';
  left_justified_conv.flags =
      LIBC_NAMESPACE::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_ptr = const_cast<char *>("ghi");
  LIBC_NAMESPACE::printf_core::convert(&writer, left_justified_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "ghi ");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcPrintfConverterTest, IntConversionSimple) {
  LIBC_NAMESPACE::printf_core::FormatSection section;
  section.has_conv = true;
  section.raw_string = "%d";
  section.conv_name = 'd';
  section.conv_val_raw = 12345;
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "12345");
  ASSERT_EQ(writer.get_chars_written(), 5);
}

TEST_F(LlvmLibcPrintfConverterTest, HexConversion) {
  LIBC_NAMESPACE::printf_core::FormatSection section;
  section.has_conv = true;
  section.raw_string = "%#018x";
  section.conv_name = 'x';
  section.flags = static_cast<LIBC_NAMESPACE::printf_core::FormatFlags>(
      LIBC_NAMESPACE::printf_core::FormatFlags::ALTERNATE_FORM |
      LIBC_NAMESPACE::printf_core::FormatFlags::LEADING_ZEROES);
  section.min_width = 18;
  section.conv_val_raw = 0x123456ab;
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = '\0';
  ASSERT_STREQ(str, "0x00000000123456ab");
  ASSERT_EQ(writer.get_chars_written(), 18);
}

TEST_F(LlvmLibcPrintfConverterTest, BinaryConversion) {
  LIBC_NAMESPACE::printf_core::FormatSection section;
  section.has_conv = true;
  section.raw_string = "%b";
  section.conv_name = 'b';
  section.conv_val_raw = 42;
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "101010");
  ASSERT_EQ(writer.get_chars_written(), 6);
}

TEST_F(LlvmLibcPrintfConverterTest, PointerConversion) {

  LIBC_NAMESPACE::printf_core::FormatSection section;
  section.has_conv = true;
  section.raw_string = "%p";
  section.conv_name = 'p';
  section.conv_val_ptr = (void *)(0x123456ab);
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = '\0';
  ASSERT_STREQ(str, "0x123456ab");
  ASSERT_EQ(writer.get_chars_written(), 10);
}

TEST_F(LlvmLibcPrintfConverterTest, OctConversion) {

  LIBC_NAMESPACE::printf_core::FormatSection section;
  section.has_conv = true;
  section.raw_string = "%o";
  section.conv_name = 'o';
  section.conv_val_raw = 01234;
  LIBC_NAMESPACE::printf_core::convert(&writer, section);

  wb.buff[wb.buff_cur] = '\0';
  ASSERT_STREQ(str, "1234");
  ASSERT_EQ(writer.get_chars_written(), 4);
}
