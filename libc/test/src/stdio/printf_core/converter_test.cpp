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

TEST(LlvmLibcPrintfConverterTest, SimpleRawConversion) {
  char str[10];
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer(
      reinterpret_cast<void *>(&str_writer),
      __llvm_libc::printf_core::write_to_string);

  __llvm_libc::printf_core::FormatSection raw_section;
  raw_section.has_conv = false;
  raw_section.raw_string = "abc";
  raw_section.raw_len = 3;

  __llvm_libc::printf_core::convert(&writer, raw_section);

  str_writer.terminate();

  ASSERT_STREQ(str, "abc");
  ASSERT_EQ(writer.get_chars_written(), 3ull);
}

TEST(LlvmLibcPrintfConverterTest, PercentConversion) {
  char str[20];
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer(
      reinterpret_cast<void *>(&str_writer),
      __llvm_libc::printf_core::write_to_string);

  __llvm_libc::printf_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "abc123";
  simple_conv.raw_len = 6;
  simple_conv.conv_name = '%';

  __llvm_libc::printf_core::convert(&writer, simple_conv);

  str[1] = '\0';

  ASSERT_STREQ(str, "%");
  ASSERT_EQ(writer.get_chars_written(), 1ull);
}

TEST(LlvmLibcPrintfConverterTest, CharConversion) {
  char str[20];
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer(
      reinterpret_cast<void *>(&str_writer),
      __llvm_libc::printf_core::write_to_string);

  __llvm_libc::printf_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "abc123";
  simple_conv.raw_len = 6;
  simple_conv.conv_name = 'c';
  simple_conv.conv_val_raw = 'D';

  __llvm_libc::printf_core::convert(&writer, simple_conv);

  str[1] = '\0';

  ASSERT_STREQ(str, "D");
  ASSERT_EQ(writer.get_chars_written(), 1ull);

  __llvm_libc::printf_core::FormatSection right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = "abc123";
  right_justified_conv.raw_len = 6;
  right_justified_conv.conv_name = 'c';
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_raw = 'E';
  __llvm_libc::printf_core::convert(&writer, right_justified_conv);

  str[5] = '\0';

  ASSERT_STREQ(str, "D   E");
  ASSERT_EQ(writer.get_chars_written(), 5ull);

  __llvm_libc::printf_core::FormatSection left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = "abc123";
  left_justified_conv.raw_len = 6;
  left_justified_conv.conv_name = 'c';
  left_justified_conv.flags =
      __llvm_libc::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_raw = 'F';
  __llvm_libc::printf_core::convert(&writer, left_justified_conv);

  str[9] = '\0';

  ASSERT_STREQ(str, "D   EF   ");
  ASSERT_EQ(writer.get_chars_written(), 9ull);
}

TEST(LlvmLibcPrintfConverterTest, StringConversion) {
  char str[20];
  __llvm_libc::printf_core::StringWriter str_writer(str);
  __llvm_libc::printf_core::Writer writer(
      reinterpret_cast<void *>(&str_writer),
      __llvm_libc::printf_core::write_to_string);

  __llvm_libc::printf_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "abc123";
  simple_conv.raw_len = 6;
  simple_conv.conv_name = 's';
  simple_conv.conv_val_ptr = const_cast<char *>("DEF");

  __llvm_libc::printf_core::convert(&writer, simple_conv);

  str[3] = '\0'; // this null terminator is just for checking after every step.

  ASSERT_STREQ(str, "DEF");
  ASSERT_EQ(writer.get_chars_written(), 3ull);

  // continuing to write to this str_writer will overwrite that null terminator.

  __llvm_libc::printf_core::FormatSection high_precision_conv;
  high_precision_conv.has_conv = true;
  high_precision_conv.raw_string = "abc123";
  high_precision_conv.raw_len = 6;
  high_precision_conv.conv_name = 's';
  high_precision_conv.precision = 4;
  high_precision_conv.conv_val_ptr = const_cast<char *>("456");
  __llvm_libc::printf_core::convert(&writer, high_precision_conv);

  str[6] = '\0';

  ASSERT_STREQ(str, "DEF456");
  ASSERT_EQ(writer.get_chars_written(), 6ull);

  __llvm_libc::printf_core::FormatSection low_precision_conv;
  low_precision_conv.has_conv = true;
  low_precision_conv.raw_string = "abc123";
  low_precision_conv.raw_len = 6;
  low_precision_conv.conv_name = 's';
  low_precision_conv.precision = 2;
  low_precision_conv.conv_val_ptr = const_cast<char *>("xyz");
  __llvm_libc::printf_core::convert(&writer, low_precision_conv);

  str[8] = '\0';

  ASSERT_STREQ(str, "DEF456xy");
  ASSERT_EQ(writer.get_chars_written(), 8ull);

  __llvm_libc::printf_core::FormatSection right_justified_conv;
  right_justified_conv.has_conv = true;
  right_justified_conv.raw_string = "abc123";
  right_justified_conv.raw_len = 6;
  right_justified_conv.conv_name = 's';
  right_justified_conv.min_width = 4;
  right_justified_conv.conv_val_ptr = const_cast<char *>("789");
  __llvm_libc::printf_core::convert(&writer, right_justified_conv);

  str[12] = '\0';

  ASSERT_STREQ(str, "DEF456xy 789");
  ASSERT_EQ(writer.get_chars_written(), 12ull);

  __llvm_libc::printf_core::FormatSection left_justified_conv;
  left_justified_conv.has_conv = true;
  left_justified_conv.raw_string = "abc123";
  left_justified_conv.raw_len = 6;
  left_justified_conv.conv_name = 's';
  left_justified_conv.flags =
      __llvm_libc::printf_core::FormatFlags::LEFT_JUSTIFIED;
  left_justified_conv.min_width = 4;
  left_justified_conv.conv_val_ptr = const_cast<char *>("ghi");
  __llvm_libc::printf_core::convert(&writer, left_justified_conv);

  str[16] = '\0';

  ASSERT_STREQ(str, "DEF456xy 789ghi ");
  ASSERT_EQ(writer.get_chars_written(), 16ull);
}
