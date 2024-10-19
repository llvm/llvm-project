//===-- Unittests for the printf Converter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/converter.h"
#include "src/time/strftime_core/core_structs.h"
#include "test/UnitTest/Test.h"
#include <time.h>

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
  tm time;
  time.tm_wday = 1;
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%a";
  simple_conv.conv_name = 'a';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "Mon");
  ASSERT_EQ(writer.get_chars_written(), 3);
}
TEST_F(LlvmLibcStrftimeConverterTest, AbbreviatedMonthNameConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  tm time;
  time.tm_mon = 4; // May
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
  tm time;
  time.tm_year = 122; // Represents 2022
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
  tm time;
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
  tm time;
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
  tm time;
  time.tm_mon = 4; // May
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
  tm time;
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
  tm time;
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
  tm time;
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
  tm time;
  time.tm_hour = 14; // 2 PM
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
  tm time;
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
  tm time;
  time.tm_year = 122; // Represents 2022 (1900 + 122)
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
  tm time;
  time.tm_year = 122; // Represents 2022 (1900 + 122)
  simple_conv.has_conv = true;
  simple_conv.raw_string = "%y";
  simple_conv.conv_name = 'y';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  ASSERT_STREQ(str, "22");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, IsoYearEdgeCaseConversionEndOfYear) {
  LIBC_NAMESPACE::strftime_core::FormatSection iso_year;
  tm time;

  // Set up the time for December 31, 2022 (a Saturday)
  time.tm_year = 122; // Represents 2022
  time.tm_mon = 11;   // December (0-based)
  time.tm_mday = 31;  // 31st day of the month
  time.tm_wday = 6;   // Saturday (0-based, 6 is Saturday)

  iso_year.has_conv = true;
  iso_year.raw_string = "%G";
  iso_year.conv_name = 'G';
  iso_year.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, iso_year);

  wb.buff[wb.buff_cur] = '\0';

  // The ISO year for this date is 2021
  ASSERT_STREQ(str, "2021");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcStrftimeConverterTest, IsoYearEdgeCaseConversionStartOfYear) {
  LIBC_NAMESPACE::strftime_core::FormatSection iso_year;
  tm time;

  // Set up the time for January 1, 2023 (a Sunday)
  time.tm_year = 123; // Represents 2023
  time.tm_mon = 0;    // January (0-based)
  time.tm_mday = 1;   // 1st day of the month
  time.tm_wday = 0;   // Sunday (0-based, 0 is Sunday)

  iso_year.has_conv = true;
  iso_year.raw_string = "%G";
  iso_year.conv_name = 'G';
  iso_year.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, iso_year);

  wb.buff[wb.buff_cur] = '\0';

  // The ISO year for this date is 2022, not 2023
  ASSERT_STREQ(str, "2022");
  ASSERT_EQ(writer.get_chars_written(), 4);
}

TEST_F(LlvmLibcStrftimeConverterTest, WeekNumberSundayFirstDayConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  tm time;

  // Set the time for a known Sunday (2023-05-07, which is a Sunday)
  time.tm_year = 123; // Represents 2023
  time.tm_mon = 4;    // May (0-based)
  time.tm_mday = 7;   // 7th day of the month
  time.tm_wday = 0;   // Sunday (0-based, 0 is Sunday)
  time.tm_yday = 126; // 126th day of the year

  simple_conv.has_conv = true;
  simple_conv.raw_string = "%U"; // Week number (Sunday is first day of week)
  simple_conv.conv_name = 'U';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  // The week number for May 7, 2023 (Sunday as first day) should be 19
  ASSERT_STREQ(str, "19");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, WeekNumberMondayFirstDayConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  tm time;

  // Set the time for a known Monday (2023-05-08, which is a Monday)
  time.tm_year = 123; // Represents 2023
  time.tm_mon = 4;    // May (0-based)
  time.tm_mday = 8;   // 8th day of the month
  time.tm_wday = 1;   // Monday (0-based, 1 is Monday)
  time.tm_yday = 127; // 127th day of the year

  simple_conv.has_conv = true;
  simple_conv.raw_string = "%W"; // Week number (Monday is first day of week)
  simple_conv.conv_name = 'W';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  // The week number for May 8, 2023 (Monday as first day) should be 19
  ASSERT_STREQ(str, "19");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, ISO8601WeekNumberConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  tm time;

  // Set the time for a known date (2023-05-10)
  time.tm_year = 123; // Represents 2023
  time.tm_mon = 4;    // May (0-based)
  time.tm_mday = 10;  // 10th day of the month
  time.tm_wday = 3;   // Wednesday (0-based, 3 is Wednesday)
  time.tm_yday = 129; // 129th day of the year

  simple_conv.has_conv = true;
  simple_conv.raw_string = "%V"; // ISO 8601 week number
  simple_conv.conv_name = 'V';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  // The ISO week number for May 10, 2023 should be 19
  ASSERT_STREQ(str, "19");
  ASSERT_EQ(writer.get_chars_written(), 2);
}

TEST_F(LlvmLibcStrftimeConverterTest, DayOfYearConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  tm time;

  // Set the time for a known date (2023-02-25)
  time.tm_year = 123; // Represents 2023
  time.tm_mon = 1;    // February (0-based)
  time.tm_mday = 25;  // 25th day of the month
  time.tm_yday = 55;  // 55th day of the year

  simple_conv.has_conv = true;
  simple_conv.raw_string = "%j"; // Day of the year
  simple_conv.conv_name = 'j';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  // The day of the year for February 25, 2023 should be 056 (padded)
  ASSERT_STREQ(str, "056");
  ASSERT_EQ(writer.get_chars_written(), 3);
}

TEST_F(LlvmLibcStrftimeConverterTest, ISO8601DateConversion) {
  LIBC_NAMESPACE::strftime_core::FormatSection simple_conv;
  tm time;

  // Set the time for a known date (2023-05-10)
  time.tm_year = 123; // Represents 2023
  time.tm_mon = 4;    // May (0-based)
  time.tm_mday = 10;  // 10th day of the month

  simple_conv.has_conv = true;
  simple_conv.raw_string = "%F"; // ISO 8601 date format
  simple_conv.conv_name = 'F';
  simple_conv.time = &time;

  LIBC_NAMESPACE::strftime_core::convert(&writer, simple_conv);

  wb.buff[wb.buff_cur] = '\0';

  // The ISO date format for May 10, 2023 should be 2023-05-10
  ASSERT_STREQ(str, "2023-05-10");
  ASSERT_EQ(writer.get_chars_written(), 10);
}
