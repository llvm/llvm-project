//===-- Format specifier converter for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See htto_conv.times://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CONVERTER_H

#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/core_structs.h"
#include "src/time/strftime_core/time_internal_def.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

#define RET_IF_RESULT_NEGATIVE(func)                                           \
  {                                                                            \
    int result = (func);                                                       \
    if (result < 0)                                                            \
      return result;                                                           \
  }

namespace details {

LIBC_INLINE cpp::optional<cpp::string_view>
num_to_strview(uintmax_t num, cpp::span<char> bufref) {
  return IntegerToString<uintmax_t>::format_to(bufref, num);
}

template <int width>
LIBC_INLINE int write_num(uintmax_t num, printf_core::Writer *writer) {
  cpp::array<char, width> buf;
  return writer->write(*num_to_strview(num, buf));
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

template <int width, char padding>
LIBC_INLINE int write_num_with_padding(uintmax_t num,
                                       printf_core::Writer *writer) {
  cpp::array<char, width> buf;
  int digits = count_digits(num);
  int padding_needed = width - digits;

  for (int _ = 0; _ < padding_needed; _++) {
    RET_IF_RESULT_NEGATIVE(writer->write(padding));
  }

  return writer->write(*num_to_strview(num, buf));
}

} // namespace details

/* Nonzero if YEAR is a leap year (every 4 years,
   except every 100th isn't, and every 400th is).  */
LIBC_INLINE bool is_leap(int year) {
  return ((year % 4 == 0) && (year % 100 != 0 || year % 400 == 0));
}

// Conversion functions

LIBC_INLINE int convert_weekday(printf_core::Writer *writer,
                                const FormatSection &to_conv) {
  return writer->write(safe_day_name(to_conv.time->tm_wday));
}

LIBC_INLINE int convert_abbreviated_weekday(printf_core::Writer *writer,
                                            const FormatSection &to_conv) {
  return writer->write(safe_abbreviated_day_name(to_conv.time->tm_wday));
}

LIBC_INLINE int convert_zero_padded_day_of_year(printf_core::Writer *writer,
                                                const FormatSection &to_conv) {
  return details::write_num_with_padding<3, '0'>(to_conv.time->tm_yday + 1,
                                                 writer);
}

LIBC_INLINE int convert_zero_padded_day_of_month(printf_core::Writer *writer,
                                                 const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>(to_conv.time->tm_mday, writer);
}

LIBC_INLINE int
convert_space_padded_day_of_month(printf_core::Writer *writer,
                                  const FormatSection &to_conv) {
  return details::write_num_with_padding<2, ' '>(to_conv.time->tm_mday, writer);
}

LIBC_INLINE int convert_decimal_weekday(printf_core::Writer *writer,
                                        const FormatSection &to_conv) {
  return details::write_num<1>(
      to_conv.time->tm_wday == 0 ? 7 : to_conv.time->tm_wday, writer);
}

LIBC_INLINE int convert_decimal_weekday_iso(printf_core::Writer *writer,
                                            const FormatSection &to_conv) {
  return details::write_num<1>(to_conv.time->tm_wday, writer);
}

LIBC_INLINE int convert_week_number_sunday(printf_core::Writer *writer,
                                           const FormatSection &to_conv) {
  int wday = to_conv.time->tm_wday;
  int yday = to_conv.time->tm_yday;
  int week = (yday - wday + 7) / 7;
  return details::write_num_with_padding<2, '0'>(week, writer);
}

LIBC_INLINE int convert_week_number_monday(printf_core::Writer *writer,
                                           const FormatSection &to_conv) {
  int wday = (to_conv.time->tm_wday + 6) % 7;
  int yday = to_conv.time->tm_yday;
  int week = (yday - wday + 7) / 7;
  return details::write_num_with_padding<2, '0'>(week, writer);
}

LIBC_INLINE int convert_full_month(printf_core::Writer *writer,
                                   const FormatSection &to_conv) {
  return writer->write(safe_month_name(to_conv.time->tm_mon));
}

LIBC_INLINE int convert_abbreviated_month(printf_core::Writer *writer,
                                          const FormatSection &to_conv) {
  return writer->write(safe_abbreviated_month_name(to_conv.time->tm_mon));
}

LIBC_INLINE int convert_zero_padded_month(printf_core::Writer *writer,
                                          const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>(to_conv.time->tm_mon + 1,
                                                 writer);
}

LIBC_INLINE int convert_full_year(printf_core::Writer *writer,
                                  const FormatSection &to_conv) {
  return details::write_num_with_padding<4, '0'>(to_conv.time->tm_year + 1900,
                                                 writer);
}

LIBC_INLINE int convert_two_digit_year(printf_core::Writer *writer,
                                       const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>(
      (to_conv.time->tm_year + 1900) % 100, writer);
}

LIBC_INLINE int convert_century(printf_core::Writer *writer,
                                const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>(
      (to_conv.time->tm_year + 1900) / 100, writer);
}

LIBC_INLINE int convert_hour_24(printf_core::Writer *writer,
                                const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>(to_conv.time->tm_hour, writer);
}

LIBC_INLINE int convert_pm(printf_core::Writer *writer,
                           const FormatSection &to_conv) {
  static const cpp::string_view AM = "AM";
  static const cpp::string_view PM = "PM";
  return writer->write(to_conv.time->tm_hour >= 12 ? PM : AM);
}

LIBC_INLINE int convert_hour_12(printf_core::Writer *writer,
                                const FormatSection &to_conv) {
  int hour = to_conv.time->tm_hour % 12;
  if (hour == 0)
    hour = 12;
  return details::write_num_with_padding<2, '0'>(hour, writer);
}

LIBC_INLINE int convert_minute(printf_core::Writer *writer,
                               const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>(to_conv.time->tm_min, writer);
}

LIBC_INLINE int convert_second(printf_core::Writer *writer,
                               const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>(to_conv.time->tm_sec, writer);
}

LIBC_INLINE int convert_MM_DD_YY(printf_core::Writer *writer,
                                 const FormatSection &to_conv) {
  RET_IF_RESULT_NEGATIVE(convert_zero_padded_month(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write("/"));
  RET_IF_RESULT_NEGATIVE(convert_zero_padded_day_of_month(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write("/"));
  return convert_two_digit_year(writer, to_conv);
}

LIBC_INLINE int convert_YYYY_MM_DD(printf_core::Writer *writer,
                                   const FormatSection &to_conv) {
  RET_IF_RESULT_NEGATIVE(convert_full_year(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write("-"));
  RET_IF_RESULT_NEGATIVE(convert_zero_padded_month(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write("-"));
  return convert_zero_padded_day_of_month(writer, to_conv);
}

LIBC_INLINE int convert_hour_minute(printf_core::Writer *writer,
                                    const FormatSection &to_conv) {
  RET_IF_RESULT_NEGATIVE(convert_hour_24(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(":"));
  return convert_minute(writer, to_conv);
}

LIBC_INLINE int convert_date_time(printf_core::Writer *writer,
                                  const FormatSection &to_conv) {
  RET_IF_RESULT_NEGATIVE(convert_abbreviated_weekday(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(" "));
  RET_IF_RESULT_NEGATIVE(convert_abbreviated_month(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(" "));
  RET_IF_RESULT_NEGATIVE(convert_space_padded_day_of_month(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(" "));
  RET_IF_RESULT_NEGATIVE(convert_full_year(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(" "));
  RET_IF_RESULT_NEGATIVE(convert_hour_24(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(":"));
  RET_IF_RESULT_NEGATIVE(convert_minute(writer, to_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(":"));
  return convert_second(writer, to_conv);
}

LIBC_INLINE int convert_time_zone(printf_core::Writer *writer) {
  return writer->write("UTC");
}

LIBC_INLINE int convert_time_zone_offset(printf_core::Writer *writer) {
  return writer->write("+0000");
}

static int iso_week_days(int yday, int wday) {
  /* Add enough to the first operand of % to make it nonnegative.  */
  int big_enough_multiple_of_7 = (-YDAY_MINIMUM / 7 + 2) * 7;
  return (yday - (yday - wday + ISO_WEEK1_WDAY + big_enough_multiple_of_7) % 7 +
          ISO_WEEK1_WDAY - ISO_WEEK_START_WDAY);
}

LIBC_INLINE int convert_iso_year(printf_core::Writer *writer,
                                 const FormatSection &to_conv) {
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

  return details::write_num<4>(year, writer);
}

LIBC_INLINE int convert_iso_day(printf_core::Writer *writer,
                                const FormatSection &to_conv) {
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
  return details::write_num_with_padding<2, '0'>(days / 7 + 1, writer);
}

int convert(printf_core::Writer *writer, const FormatSection &to_conv) {
  if (!to_conv.has_conv)
    return writer->write(to_conv.raw_string);
  switch (to_conv.conv_name) {
  case '%':
    return writer->write("%");
  // day of the week
  case 'A':
    return convert_weekday(writer, to_conv);
  case 'a':
    return convert_abbreviated_weekday(writer, to_conv);
  case 'w':
    return convert_decimal_weekday(writer, to_conv);
  case 'u':
    return convert_decimal_weekday_iso(writer, to_conv);
  // day of the year/ month
  case 'j':
    return convert_zero_padded_day_of_year(writer, to_conv);
  case 'd':
    return convert_zero_padded_day_of_month(writer, to_conv);
  case 'e':
    return convert_space_padded_day_of_month(writer, to_conv);
  // week
  case 'U':
    return convert_week_number_sunday(writer, to_conv);
  case 'W':
    return convert_week_number_monday(writer, to_conv);
  case 'V':
    return convert_iso_day(writer, to_conv);
  // month
  case 'B':
    return convert_full_month(writer, to_conv);
  case 'b':
  case 'h':
    return convert_abbreviated_month(writer, to_conv);
  case 'm':
    return convert_zero_padded_month(writer, to_conv);
  // year
  case 'Y':
    return convert_full_year(writer, to_conv);
  case 'y':
    return convert_two_digit_year(writer, to_conv);
  case 'C':
    return convert_century(writer, to_conv);
  case 'G':
    return convert_iso_year(writer, to_conv);
  // hours
  case 'p':
    return convert_pm(writer, to_conv);
  case 'H':
    return convert_hour_24(writer, to_conv);
  case 'I':
    return convert_hour_12(writer, to_conv);
  // minutes
  case 'M':
    return convert_minute(writer, to_conv);
  // seconds
  case 'S':
    return convert_second(writer, to_conv);
    // Date and Time
  case 'c':
    return convert_date_time(writer, to_conv);

  // Timezone
  case 'Z':
    return convert_time_zone(writer);
  case 'z':
    return convert_time_zone_offset(writer);

  // Custom formats: MM/DD/YY, YYYY-MM-DD
  case 'D':
    return convert_MM_DD_YY(writer, to_conv); // Equivalent to %m/%d/%y
  case 'F':
    return convert_YYYY_MM_DD(writer, to_conv); // Equivalent to %Y-%m-%d

  // 24-hour time without seconds (HH:MM)
  case 'R':
    return convert_hour_minute(writer, to_conv); // Equivalent to %H:%M
  default:
    writer->write(to_conv.raw_string);
  }
  return 0;
}

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL
#endif
