//===-- Format specifier converter for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See htto_conv.times://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_NUM_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_NUM_CONVERTER_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/core_structs.h"
#include "src/time/strftime_core/time_internal_def.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

namespace details {

LIBC_INLINE cpp::optional<cpp::string_view>
num_to_strview(uintmax_t num, cpp::span<char> bufref) {
  return IntegerToString<uintmax_t>::format_to(bufref, num);
}

template <typename T> int count_digits(T num) {
  if (num == 0)
    return 1;
  int digits = 0;
  while (num > 0) {
    num /= 10;
    digits++;
  }
  return digits;
}

LIBC_INLINE int write_num_with_padding(int width, char padding, uintmax_t num,
                                       printf_core::Writer *writer) {
  cpp::array<char, IntegerToString<uintmax_t>::buffer_size()> buf;
  int digits = count_digits(num);
  int padding_needed = width - digits;

  for (int _ = 0; _ < padding_needed; _++) {
    RET_IF_RESULT_NEGATIVE(writer->write(padding));
  }

  return writer->write(*num_to_strview(num, buf));
}

} // namespace details

namespace iso {

/* Nonzero if YEAR is a leap year (every 4 years,
   except every 100th isn't, and every 400th is).  */
LIBC_INLINE bool is_leap(int year) {
  return ((year % 4 == 0) && (year % 100 != 0 || year % 400 == 0));
}

static int iso_week_days(int yday, int wday) {
  /* Add enough to the first operand of % to make it nonnegative.  */
  int big_enough_multiple_of_7 = (-YDAY_MINIMUM / 7 + 2) * 7;
  return (yday - (yday - wday + ISO_WEEK1_WDAY + big_enough_multiple_of_7) % 7 +
          ISO_WEEK1_WDAY - ISO_WEEK_START_WDAY);
}

enum class IsoData {
  GET_DATE,
  GET_YEAR,
};

template <IsoData get_date_or_year>
LIBC_INLINE int convert_iso(const FormatSection &to_conv) {
  int year = to_conv.time->tm_year + YEAR_BASE;
  int days = iso_week_days(to_conv.time->tm_yday, to_conv.time->tm_wday);

  if (days < 0) {
    /* This ISO week belongs to the previous year.  */
    year--;
    days = iso_week_days(to_conv.time->tm_yday + (365 + is_leap(year)),
                         to_conv.time->tm_wday);
  } else {
    int d = iso_week_days(to_conv.time->tm_yday - (365 + is_leap(year)),
                          to_conv.time->tm_wday);
    if (0 <= d) {
      /* This ISO week belongs to the next year.  */
      year++;
      days = d;
    }
  }

  if constexpr (get_date_or_year == IsoData::GET_YEAR) {
    return year;
  } else {
    return days / 7 + 1;
  }
}
} // namespace iso

int write_num(printf_core::Writer *writer, const FormatSection &to_conv) {
  int num = 0;
  auto &time = *to_conv.time;

  // Handle numeric conversions based on the format specifier (conv_name)
  switch (to_conv.conv_name) {
  // Century (C) - the first two digits of the year
  case 'C':
    num = (time.tm_year + 1900) / 100;
    break;

  // Full year (Y) - the full four-digit year
  case 'Y':
    num = time.tm_year + 1900;
    break;

  // Two-digit year (y) - the last two digits of the year
  case 'y':
    num = (time.tm_year + 1900) % 100;
    break;

  // Day of the year (j) - the day number within the year (1-366)
  case 'j':
    num = time.tm_yday + 1;
    break;

  // Zero-padded month (m) - month as a zero-padded number (01-12)
  case 'm':
    num = time.tm_mon + 1;
    break;

  // Day of the month (d) - zero-padded day of the month (01-31)
  case 'd':
  case 'e':
    num = time.tm_mday;
    break;

  // 24-hour format (H) - zero-padded hour (00-23)
  case 'H':
    num = time.tm_hour;
    break;

  // 12-hour format (I) - zero-padded hour (01-12)
  case 'I':
    num = time.tm_hour % 12;
    if (num == 0)
      num = 12; // Convert 0 to 12 for 12-hour format
    break;

  // Minute (M) - zero-padded minute (00-59)
  case 'M':
    num = time.tm_min;
    break;

  // Second (S) - zero-padded second (00-59)
  case 'S':
    num = time.tm_sec;
    break;

  // Week number starting on Sunday (U) - week number of the year (Sunday as the
  // start of the week)
  case 'U': {
    int wday = time.tm_wday;
    num = (time.tm_yday - wday + 7) / 7;
    break;
  }

  // Week number starting on Monday (W) - week number of the year (Monday as the
  // start of the week)
  case 'W': {
    int wday = (time.tm_wday + 6) % 7; // Adjust to Monday as the first day
    num = (time.tm_yday - wday + 7) / 7;
    break;
  }

  // ISO week day (V) - week number following ISO 8601
  case 'V':
    num = iso::convert_iso<iso::IsoData::GET_DATE>(to_conv);
    break;

  case 'G':
    num = iso::convert_iso<iso::IsoData::GET_YEAR>(to_conv);
    break;
  // Decimal weekday (w) - day of the week (Sunday = 0, Monday = 1, etc.)
  case 'w':
    num = time.tm_wday;
    break;

  // ISO weekday (u) - day of the week (Monday = 1, Sunday = 7)
  case 'u':
    num = (time.tm_wday == 0) ? 7 : time.tm_wday;
    break;

  default:
    return writer->write(
        to_conv.raw_string); // Default: write raw string if no match
  }

  return details::write_num_with_padding(to_conv.min_width, to_conv.padding,
                                         num, writer);
}

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif
