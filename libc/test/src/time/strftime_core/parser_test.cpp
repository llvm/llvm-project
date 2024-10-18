//===-- Unittests for the printf Converter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/strftime_core/core_structs.h"
#include "src/time/strftime_core/parser.h"

#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

using namespace strftime_core;

class LlvmLibcStrftimeParserTest : public LIBC_NAMESPACE::testing::Test {
protected:
protected:
  // Sample time structure for testing
  struct tm test_time;

  // Setup function to initialize test_time
  void SetUp() override {
    test_time = {};          // Reset tm structure
    test_time.tm_year = 123; // Represents the year 2023
    test_time.tm_mon = 4;    // May
    test_time.tm_mday = 10;  // 10th day of the month
    test_time.tm_hour = 14;  // 2 PM
    test_time.tm_min = 30;   // 30 minutes
    test_time.tm_sec = 0;    // 0 seconds
    test_time.tm_isdst = 0;  // Daylight Saving Time flag
  }
};

TEST_F(LlvmLibcStrftimeParserTest, ParseRawSection) {
  const char *format = "Today is %Y-%m-%d";
  Parser parser(format, test_time);

  FormatSection section = parser.get_next_section();

  ASSERT_FALSE(section.has_conv); // Should be a raw section
  ASSERT_STREQ(section.raw_string.data(), "Today is "); // Raw string
}

TEST_F(LlvmLibcStrftimeParserTest, ParseConversionSection) {
  const char *format = "%Y is the year";
  Parser parser(format, test_time);

  FormatSection section = parser.get_next_section();

  ASSERT_TRUE(section.has_conv);     // Should be a conversion section
  ASSERT_EQ(section.conv_name, 'Y'); // Conversion character
  ASSERT_EQ(section.time->tm_year,
            test_time.tm_year); // Check if time is passed correctly
  ASSERT_STREQ(section.raw_string.data(),
               "Y is the year"); // Raw string part after the conversion
}

TEST_F(LlvmLibcStrftimeParserTest, ParseConversionSectionWithModifiers) {
  const char *format = "%OB";
  Parser parser(format, test_time);

  FormatSection section = parser.get_next_section();

  ASSERT_TRUE(section.has_conv);     // Should be a conversion section
  ASSERT_EQ(section.conv_name, 'B'); // Conversion character
  ASSERT_TRUE(section.isO);          // Check if the 'O' modifier was set
  ASSERT_FALSE(section.isE);         // Check if the 'E' modifier was not set
}

TEST_F(LlvmLibcStrftimeParserTest, HandleInvalidConversion) {
  const char *format = "%X"; // Invalid conversion
  Parser parser(format, test_time);

  FormatSection section = parser.get_next_section();

  ASSERT_FALSE(section.has_conv); // Should be treated as an invalid conversion
  ASSERT_STREQ(section.raw_string.data(),
               "%X"); // Raw string should be unchanged
}

TEST_F(LlvmLibcStrftimeParserTest, HandleMultipleSections) {
  const char *format = "%Y-%m-%d %H:%M:%S";
  Parser parser(format, test_time);

  // Parse year section
  FormatSection section1 = parser.get_next_section();
  ASSERT_TRUE(section1.has_conv);
  ASSERT_EQ(section1.conv_name, 'Y');

  // Parse separator
  FormatSection section2 = parser.get_next_section();
  ASSERT_FALSE(section2.has_conv);
  ASSERT_STREQ(section2.raw_string.data(), "-");

  // Parse month section
  FormatSection section3 = parser.get_next_section();
  ASSERT_TRUE(section3.has_conv);
  ASSERT_EQ(section3.conv_name, 'm');

  // Parse another separator
  FormatSection section4 = parser.get_next_section();
  ASSERT_FALSE(section4.has_conv);
  ASSERT_STREQ(section4.raw_string.data(), "-");

  // Parse day section
  FormatSection section5 = parser.get_next_section();
  ASSERT_TRUE(section5.has_conv);
  ASSERT_EQ(section5.conv_name, 'd');
}

} // namespace LIBC_NAMESPACE_DECL
