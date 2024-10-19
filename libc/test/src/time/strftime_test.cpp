
//===-- Unittests for ctime -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/core_structs.h"
#include "src/time/strftime_core/strftime_main.h"
#include "test/UnitTest/Test.h"
#include <time.h>

namespace LIBC_NAMESPACE_DECL {

using namespace strftime_core;
size_t call_strftime(char *__restrict buffer, size_t buffsz,
                     const char *__restrict format, const struct tm *timeptr) {

  printf_core::WriteBuffer wb(buffer, (buffsz > 0 ? buffsz - 1 : 0));
  printf_core::Writer writer(&wb);
  strftime_core::strftime_main(&writer, format, timeptr);
  return writer.get_chars_written();
}

TEST(LlvmLibcStrftimeTest, FormatsYearMonthDayCorrectly) {
  struct tm time;
  time.tm_year = 122; // Year since 1900, so 2022
  time.tm_mon = 9;    // October (0-indexed)
  time.tm_mday = 15;  // 15th day

  char buffer[100];
  call_strftime(buffer, sizeof(buffer), "%Y-%m-%d", &time);
  EXPECT_STREQ(buffer, "2022-10-15");
}

TEST(LlvmLibcStrftimeTest, FormatsTimeCorrectly) {
  struct tm time;
  time.tm_hour = 14; // 2:00 PM
  time.tm_min = 30;  // 30 minutes
  time.tm_sec = 45;  // 45 seconds

  char buffer[100];
  call_strftime(buffer, sizeof(buffer), "%H:%M:%S", &time);
  EXPECT_STREQ(buffer, "14:30:45");
}

TEST(LlvmLibcStrftimeTest, FormatsAmPmCorrectly) {
  struct tm time;
  time.tm_hour = 13; // 1:00 PM
  time.tm_min = 0;

  char buffer[100];
  call_strftime(buffer, sizeof(buffer), "%I:%M %p", &time);
  EXPECT_STREQ(buffer, "01:00 PM");
}

TEST(LlvmLibcStrftimeTest, HandlesLeapYear) {
  struct tm time;
  time.tm_year = 120; // Year 2020
  time.tm_mon = 1;    // February
  time.tm_mday = 29;  // 29th day

  char buffer[100];
  call_strftime(buffer, sizeof(buffer), "%Y-%m-%d", &time);
  EXPECT_STREQ(buffer, "2020-02-29");
}

TEST(LlvmLibcStrftimeTest, HandlesEndOfYear) {
  struct tm time;
  time.tm_year = 121; // Year 2021
  time.tm_mon = 11;   // December
  time.tm_mday = 31;  // 31st day

  char buffer[100];
  call_strftime(buffer, sizeof(buffer), "%Y-%m-%d", &time);
  EXPECT_STREQ(buffer, "2021-12-31");
}

TEST(LlvmLibcStrftimeTest, FormatsTimezoneCorrectly) {
  struct tm time;
  time.tm_year = 122; // Year 2022
  time.tm_mon = 9;    // October
  time.tm_mday = 15;
  time.tm_hour = 12;
  time.tm_min = 0;
  time.tm_sec = 0;
  time.tm_isdst = -1; // Use system's daylight saving time information

  char buffer[100];
  call_strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %Z", &time);
  EXPECT_STRNE(buffer, "");
}

} // namespace LIBC_NAMESPACE_DECL
