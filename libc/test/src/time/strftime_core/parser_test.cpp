//===-- Unittests for the printf Converter --------------------------------===//
//

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
  struct tm test_time;

  void SetUp() override {
    test_time = {};
    test_time.tm_year = 123;
    test_time.tm_mon = 4;
    test_time.tm_mday = 10;
    test_time.tm_hour = 14;
    test_time.tm_min = 30;
    test_time.tm_sec = 0;
    test_time.tm_isdst = 0;
  }
};

TEST_F(LlvmLibcStrftimeParserTest, ParseRawSection) {
  const char *format = "Today is %Y-%m-%d";
  Parser parser(format, test_time);

  FormatSection section = parser.get_next_section();

  ASSERT_FALSE(section.has_conv);
  ASSERT_EQ(section.raw_string, cpp::string_view("Today is "));
}

TEST_F(LlvmLibcStrftimeParserTest, ParseConversionSection) {
  const char *format = "%Y is the year";
  Parser parser(format, test_time);

  FormatSection section = parser.get_next_section();

  ASSERT_TRUE(section.has_conv);
  ASSERT_EQ(section.conv_name, 'Y');
  ASSERT_EQ(section.time->tm_year, test_time.tm_year);
  ASSERT_EQ(section.raw_string, cpp::string_view("%Y"));
}

TEST_F(LlvmLibcStrftimeParserTest, ParseConversionSectionWithModifiers) {
  const char *format = "%Od";
  Parser parser(format, test_time);

  FormatSection section = parser.get_next_section();

  ASSERT_TRUE(section.has_conv);
  ASSERT_EQ(section.conv_name, 'd');
  ASSERT_TRUE(section.isO);
  ASSERT_FALSE(section.isE);
}

TEST_F(LlvmLibcStrftimeParserTest, HandleInvalidConversion) {
  const char *format = "%k";
  Parser parser(format, test_time);

  FormatSection section = parser.get_next_section();

  ASSERT_FALSE(section.has_conv);
  ASSERT_EQ(section.raw_string, cpp::string_view("%k"));
}

TEST_F(LlvmLibcStrftimeParserTest, HandleMultipleSections) {
  const char *format = "%Y-%m-%d %H:%M:%S";
  Parser parser(format, test_time);

  FormatSection section1 = parser.get_next_section();
  ASSERT_TRUE(section1.has_conv);
  ASSERT_EQ(section1.conv_name, 'Y');

  FormatSection section2 = parser.get_next_section();
  ASSERT_FALSE(section2.has_conv);
  ASSERT_EQ(section2.raw_string, cpp::string_view("-"));

  FormatSection section3 = parser.get_next_section();
  ASSERT_TRUE(section3.has_conv);
  ASSERT_EQ(section3.conv_name, 'm');

  FormatSection section4 = parser.get_next_section();
  ASSERT_FALSE(section4.has_conv);
  ASSERT_EQ(section4.raw_string, cpp::string_view("-"));

  FormatSection section5 = parser.get_next_section();
  ASSERT_TRUE(section5.has_conv);
  ASSERT_EQ(section5.conv_name, 'd');
}

} // namespace LIBC_NAMESPACE_DECL
