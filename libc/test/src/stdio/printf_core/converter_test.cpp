//===-- Unittests for the printf Converter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/converter.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/string_writer.h"
#include "src/stdio/printf_core/writer.h"

#include "utils/UnitTest/Test.h"

class LlvmLibcPrintfConverterTest : public __llvm_libc::testing::Test {
protected:
  // void SetUp() override {}
  // void TearDown() override {}

  char str[60];
  __llvm_libc::printf_core::StringWriter str_writer =
      __llvm_libc::printf_core::StringWriter(str);
  __llvm_libc::printf_core::Writer writer = __llvm_libc::printf_core::Writer(
      reinterpret_cast<void *>(&str_writer),
      __llvm_libc::printf_core::StringWriter::write_str,
      __llvm_libc::printf_core::StringWriter::write_chars,
      __llvm_libc::printf_core::StringWriter::write_char);
};

TEST_F(LlvmLibcPrintfConverterTest, SimpleRawConversion) {
  __llvm_libc::printf_core::FormatSection raw_section;
  raw_section.has_conv = false;
  raw_section.raw_string = "abc";

  __llvm_libc::printf_core::convert(&writer, raw_section);

  str_writer.terminate();

  ASSERT_STREQ(str, "abc");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcPrintfConverterTest, PercentConversion) {
  __llvm_libc::printf_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%%";
  simple_conv.conv_name = '%';

  __llvm_libc::printf_core::convert(&writer, simple_conv);

  str[1] = '\0';

  ASSERT_STREQ(str, "%");
  ASSERT_EQ(writer.get_chars_written(), 1);
}

TEST_F(LlvmLibcPrintfConverterTest, CharConversionSimple) {
  __llvm_libc::printf_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  // If has_conv is true, the raw string is ignored. They are not being parsed
  // and match the actual conversion taking place so that you can compare these
  // tests with other implmentations. The raw strings are completely optional.
  simple_conv.raw_string = "%c";
  simple_conv.conv_name = 'c';
  simple_conv.conv_val_raw = 'D';

  __llvm_libc::printf_core::convert(&writer, simple_conv);

  str_writer.terminate();

  ASSERT_STREQ(str, "D");
  ASSERT_EQ(writer.get_chars_written(), 1);
}

TEST_F(LlvmLibcPrintfConverterTest, CharConversionRightJustified) {
  __llvm_libc::printf_core::FormatSection right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = "%4c";
  right_justified_conv.conv_name = 'c';
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_raw = 'E';
  __llvm_libc::printf_core::convert(&writer, right_justified_conv);

  str_writer.terminate();

  ASSERT_STREQ(str, "   E");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcPrintfConverterTest, CharConversionLeftJustified) {
  __llvm_libc::printf_core::FormatSection left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = "%-4c";
  left_justified_conv.conv_name = 'c';
  left_justified_conv.flags =
      __llvm_libc::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_raw = 'F';
  __llvm_libc::printf_core::convert(&writer, left_justified_conv);

  str_writer.terminate();

  ASSERT_STREQ(str, "F   ");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionSimple) {

  __llvm_libc::printf_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%s";
  simple_conv.conv_name = 's';
  simple_conv.conv_val_ptr = const_cast<char *>("DEF");

  __llvm_libc::printf_core::convert(&writer, simple_conv);

  str_writer.terminate();

  ASSERT_STREQ(str, "DEF");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionPrecisionHigh) {
  __llvm_libc::printf_core::FormatSection high_precision_conv;
  high_precision_conv.has_conv = true;
  high_precision_conv.raw_string = "%4s";
  high_precision_conv.conv_name = 's';
  high_precision_conv.precision = 4;
  high_precision_conv.conv_val_ptr = const_cast<char *>("456");
  __llvm_libc::printf_core::convert(&writer, high_precision_conv);

  str_writer.terminate();

  ASSERT_STREQ(str, "456");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionPrecisionLow) {
  __llvm_libc::printf_core::FormatSection low_precision_conv;
  low_precision_conv.has_conv = true;
  low_precision_conv.raw_string = "%.2s";
  low_precision_conv.conv_name = 's';
  low_precision_conv.precision = 2;
  low_precision_conv.conv_val_ptr = const_cast<char *>("xyz");
  __llvm_libc::printf_core::convert(&writer, low_precision_conv);

  str_writer.terminate();

  ASSERT_STREQ(str, "xy");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionRightJustified) {
  __llvm_libc::printf_core::FormatSection right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = "%4s";
  right_justified_conv.conv_name = 's';
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_ptr = const_cast<char *>("789");
  __llvm_libc::printf_core::convert(&writer, right_justified_conv);

  str_writer.terminate();

  ASSERT_STREQ(str, " 789");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcPrintfConverterTest, StringConversionLeftJustified) {
  __llvm_libc::printf_core::FormatSection left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = "%-4s";
  left_justified_conv.conv_name = 's';
  left_justified_conv.flags =
      __llvm_libc::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_ptr = const_cast<char *>("ghi");
  __llvm_libc::printf_core::convert(&writer, left_justified_conv);

  str_writer.terminate();

  ASSERT_STREQ(str, "ghi ");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcPrintfConverterTest, IntConversionSimple) {
  __llvm_libc::printf_core::FormatSection section;
  section.has_conv = true;
  section.raw_string = "%d";
  section.conv_name = 'd';
  section.conv_val_raw = 12345;
  __llvm_libc::printf_core::convert(&writer, section);

  str_writer.terminate();

  ASSERT_STREQ(str, "12345");
  ASSERT_EQ(writer.get_chars_written(), 5);
}

TEST_F(LlvmLibcPrintfConverterTest, HexConversion) {
  __llvm_libc::printf_core::FormatSection section;
  section.has_conv = true;
  section.raw_string = "%#018x";
  section.conv_name = 'x';
  section.flags = static_cast<__llvm_libc::printf_core::FormatFlags>(
      __llvm_libc::printf_core::FormatFlags::ALTERNATE_FORM |
      __llvm_libc::printf_core::FormatFlags::LEADING_ZEROES);
  section.min_width = 18;
  section.conv_val_raw = 0x123456ab;
  __llvm_libc::printf_core::convert(&writer, section);

  str_writer.terminate();
  ASSERT_STREQ(str, "0x00000000123456ab");
  ASSERT_EQ(writer.get_chars_written(), 18);
}

TEST_F(LlvmLibcPrintfConverterTest, PointerConversion) {

  __llvm_libc::printf_core::FormatSection section;
  section.has_conv = true;
  section.raw_string = "%p";
  section.conv_name = 'p';
  section.conv_val_ptr = (void *)(0x123456ab);
  __llvm_libc::printf_core::convert(&writer, section);

  str_writer.terminate();
  ASSERT_STREQ(str, "0x123456ab");
  ASSERT_EQ(writer.get_chars_written(), 10);
}

TEST_F(LlvmLibcPrintfConverterTest, OctConversion) {

  __llvm_libc::printf_core::FormatSection section;
  section.has_conv = true;
  section.raw_string = "%o";
  section.conv_name = 'o';
  section.conv_val_raw = 01234;
  __llvm_libc::printf_core::convert(&writer, section);

  str_writer.terminate();
  ASSERT_STREQ(str, "1234");
  ASSERT_EQ(writer.get_chars_written(), 4);
}
