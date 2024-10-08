//===-- Unittests for the printf Converter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/strftime_core/converter.h"
#include "src/stdio/strftime_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

#include "test/UnitTest/Test.h"

class LlvmLibcStrftimeConverterTest : public LIBC_NAMESPACE::testing::Test {
protected:
  // void SetUp() override {}
  // void TearDown() override {}

  char str[60];
  LIBC_NAMESPACE::printf_core::WriteBuffer wb =
      LIBC_NAMESPACE::printf_core::WriteBuffer(str, sizeof(str) - 1);
  LIBC_NAMESPACE::printf_core::Writer writer =
      LIBC_NAMESPACE::printf_core::Writer(&wb);
};

TEST_F(LlvmLibcStrftimeConverterTest, SimpleRawConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection raw_section;
  raw_section.has_conv = false;
  raw_section.raw_string = "abc";

  LIBC_NAMESPACE::strftime_core::convert(&writer, raw_section);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "abc");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcStrftimeConverterTest, PercentConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%%";
  simple_conv.conv_name = '%';

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "%");
  ASSERT_EQ(writer.get_chars_written(), 1);
}

TEST_F(LlvmLibcStrftimeConverterTest, WeekdayConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_wday = 1;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%a";
  simple_conv.conv_name = 'a';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "Monday");
  ASSERT_EQ(writer.get_chars_written(), 6);
}
TEST_F(LlvmLibcStrftimeConverterTest, AbbreviatedMonthNameConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_mon = 4;  // May
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%b";
  simple_conv.conv_name = 'b';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "May");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcStrftimeConverterTest, CenturyConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_year = 122;  // Represents 2022
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%C";
  simple_conv.conv_name = 'C';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "20");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, DayOfMonthZeroPaddedConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_mday = 7;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%d";
  simple_conv.conv_name = 'd';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "07");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, DayOfMonthSpacePaddedConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_mday = 7;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%e";
  simple_conv.conv_name = 'e';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, " 7");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, FullMonthNameConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_mon = 4;  // May
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%B";
  simple_conv.conv_name = 'B';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "May");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcStrftimeConverterTest, Hour12Conversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_hour = 14;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%I";
  simple_conv.conv_name = 'I';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "02");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, Hour24PaddedConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_hour = 9;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%H";
  simple_conv.conv_name = 'H';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "09");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, MinuteConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_min = 45;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%M";
  simple_conv.conv_name = 'M';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "45");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, AMPMConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_hour = 14;  // 2 PM
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%p";
  simple_conv.conv_name = 'p';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "PM");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, SecondsConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_sec = 30;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%S";
  simple_conv.conv_name = 'S';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "30");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, FullYearConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_year = 122;  // Represents 2022 (1900 + 122)
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%Y";
  simple_conv.conv_name = 'Y';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "2022");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcStrftimeConverterTest, TwoDigitYearConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  LIBC_NAMESPACE::strftime_core::tm time;
  time.tm_year = 122;  // Represents 2022 (1900 + 122)
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%y";
  simple_conv.conv_name = 'y';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "22");
  ASSERT_EQ(writer.get_chars_written(), 2);
}
