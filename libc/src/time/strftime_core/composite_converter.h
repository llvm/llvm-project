//===-- Composite converter for strftime ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See htto_conv.times://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_COMPOSITE_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_COMPOSITE_CONVERTER_H

#include "hdr/types/struct_tm.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/core_structs.h"
#include "src/time/strftime_core/num_converter.h"
#include "src/time/strftime_core/str_converter.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

LIBC_INLINE ErrorOr<IntFormatSection>
get_specific_int_format(const tm *timeptr, const FormatSection &base_to_conv,
                        char new_conv_name, int TRAILING_CONV_LEN = -1) {
  // a negative padding will be treated as the default
  const int NEW_MIN_WIDTH =
      TRAILING_CONV_LEN > 0 ? base_to_conv.min_width - TRAILING_CONV_LEN : 0;
  FormatSection new_conv = base_to_conv;
  new_conv.conv_name = new_conv_name;
  new_conv.min_width = NEW_MIN_WIDTH;

  auto result_or = get_int_format(new_conv, timeptr);
  if (!result_or)
    return cpp::unexpected(result_or.error());
  IntFormatSection result = result_or.value();

  // If the user set the padding, but it's below the width of the trailing
  // conversions, then there should be no padding.
  if (base_to_conv.min_width > 0 && NEW_MIN_WIDTH < 0)
    result.pad_to_len = 0;

  return result;
}

template <printf_core::WriteMode write_mode>
LIBC_INLINE int convert_date_us(printf_core::Writer<write_mode> *writer,
                                const FormatSection &to_conv,
                                const tm *timeptr) {
  // format is %m/%d/%y (month/day/year)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  constexpr int TRAILING_CONV_LEN = 1 + 2 + 1 + 2; // sizeof("/01/02")
  IntFormatSection year_conv;
  IntFormatSection mon_conv;
  IntFormatSection mday_conv;

  auto mon_conv_or =
      get_specific_int_format(timeptr, to_conv, 'm', TRAILING_CONV_LEN);
  if (!mon_conv_or)
    return -mon_conv_or.error();
  mon_conv = mon_conv_or.value();

  auto mday_conv_or = get_specific_int_format(timeptr, to_conv, 'd');
  if (!mday_conv_or)
    return -mday_conv_or.error();
  mday_conv = mday_conv_or.value();

  auto year_conv_or = get_specific_int_format(timeptr, to_conv, 'y');
  if (!year_conv_or)
    return -year_conv_or.error();
  year_conv = year_conv_or.value();

  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, mon_conv));
  RET_IF_RESULT_NEGATIVE(writer->write('/'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, mday_conv));
  RET_IF_RESULT_NEGATIVE(writer->write('/'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, year_conv));

  return WRITE_OK;
}

template <printf_core::WriteMode write_mode>
LIBC_INLINE int convert_date_iso(printf_core::Writer<write_mode> *writer,
                                 const FormatSection &to_conv,
                                 const tm *timeptr) {
  // format is "%Y-%m-%d" (year-month-day)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  constexpr int TRAILING_CONV_LEN = 1 + 2 + 1 + 2; // sizeof("-01-02")
  IntFormatSection year_conv;
  IntFormatSection mon_conv;
  IntFormatSection mday_conv;

  auto year_conv_or =
      get_specific_int_format(timeptr, to_conv, 'Y', TRAILING_CONV_LEN);
  if (!year_conv_or)
    return -year_conv_or.error();
  year_conv = year_conv_or.value();

  auto mon_conv_or = get_specific_int_format(timeptr, to_conv, 'm');
  if (!mon_conv_or)
    return -mon_conv_or.error();
  mon_conv = mon_conv_or.value();

  auto mday_conv_or = get_specific_int_format(timeptr, to_conv, 'd');
  if (!mday_conv_or)
    return -mday_conv_or.error();
  mday_conv = mday_conv_or.value();

  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, year_conv));
  RET_IF_RESULT_NEGATIVE(writer->write('-'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, mon_conv));
  RET_IF_RESULT_NEGATIVE(writer->write('-'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, mday_conv));

  return WRITE_OK;
}

template <printf_core::WriteMode write_mode>
LIBC_INLINE int convert_time_am_pm(printf_core::Writer<write_mode> *writer,
                                   const FormatSection &to_conv,
                                   const tm *timeptr) {
  // format is "%I:%M:%S %p" (hour:minute:second AM/PM)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  constexpr int TRAILING_CONV_LEN =
      1 + 2 + 1 + 2 + 1 + 2; // sizeof(":01:02 AM")
  IntFormatSection hour_conv;
  IntFormatSection min_conv;
  IntFormatSection sec_conv;

  const time_utils::TMReader time_reader(timeptr);

  auto hour_conv_or =
      get_specific_int_format(timeptr, to_conv, 'I', TRAILING_CONV_LEN);
  if (!hour_conv_or)
    return -hour_conv_or.error();
  hour_conv = hour_conv_or.value();

  auto min_conv_or = get_specific_int_format(timeptr, to_conv, 'M');
  if (!min_conv_or)
    return -min_conv_or.error();
  min_conv = min_conv_or.value();

  auto sec_conv_or = get_specific_int_format(timeptr, to_conv, 'S');
  if (!sec_conv_or)
    return -sec_conv_or.error();
  sec_conv = sec_conv_or.value();

  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, hour_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, min_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, sec_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(' '));
  RET_IF_RESULT_NEGATIVE(writer->write(time_reader.get_am_pm()));

  return WRITE_OK;
}

template <printf_core::WriteMode write_mode>
LIBC_INLINE int convert_time_minute(printf_core::Writer<write_mode> *writer,
                                    const FormatSection &to_conv,
                                    const tm *timeptr) {
  // format is "%H:%M" (hour:minute)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  constexpr int TRAILING_CONV_LEN = 1 + 2; // sizeof(":01")
  IntFormatSection hour_conv;
  IntFormatSection min_conv;

  auto hour_conv_or =
      get_specific_int_format(timeptr, to_conv, 'H', TRAILING_CONV_LEN);
  if (!hour_conv_or)
    return -hour_conv_or.error();
  hour_conv = hour_conv_or.value();

  auto min_conv_or = get_specific_int_format(timeptr, to_conv, 'M');
  if (!min_conv_or)
    return -min_conv_or.error();
  min_conv = min_conv_or.value();

  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, hour_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, min_conv));

  return WRITE_OK;
}

template <printf_core::WriteMode write_mode>
LIBC_INLINE int convert_time_second(printf_core::Writer<write_mode> *writer,
                                    const FormatSection &to_conv,
                                    const tm *timeptr) {
  // format is "%H:%M:%S" (hour:minute:second)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  constexpr int TRAILING_CONV_LEN = 1 + 2 + 1 + 2; // sizeof(":01:02")
  IntFormatSection hour_conv;
  IntFormatSection min_conv;
  IntFormatSection sec_conv;

  auto hour_conv_or =
      get_specific_int_format(timeptr, to_conv, 'H', TRAILING_CONV_LEN);
  if (!hour_conv_or)
    return -hour_conv_or.error();
  hour_conv = hour_conv_or.value();

  auto min_conv_or = get_specific_int_format(timeptr, to_conv, 'M');
  if (!min_conv_or)
    return -min_conv_or.error();
  min_conv = min_conv_or.value();

  auto sec_conv_or = get_specific_int_format(timeptr, to_conv, 'S');
  if (!sec_conv_or)
    return -sec_conv_or.error();
  sec_conv = sec_conv_or.value();

  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, hour_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, min_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, sec_conv));

  return WRITE_OK;
}

template <printf_core::WriteMode write_mode>
LIBC_INLINE int convert_full_date_time(printf_core::Writer<write_mode> *writer,
                                       const FormatSection &to_conv,
                                       const tm *timeptr) {
  const time_utils::TMReader time_reader(timeptr);
  // format is "%a %b %e %T %Y" (weekday month mday [time] year)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  // sizeof("Sun Jan 12 03:45:06 2025")
  constexpr int FULL_CONV_LEN = 3 + 1 + 3 + 1 + 2 + 1 + 8 + 1 + 4;
  // use the full conv len because this isn't being passed to a proper converter
  // that will handle the width of the leading conversion. Instead it has to be
  // handled below.
  const int requested_padding = to_conv.min_width - FULL_CONV_LEN;

  cpp::string_view wday_str = unwrap_opt(time_reader.get_weekday_short_name());
  cpp::string_view month_str = unwrap_opt(time_reader.get_month_short_name());
  IntFormatSection mday_conv;
  IntFormatSection year_conv;

  auto mday_conv_or = get_specific_int_format(timeptr, to_conv, 'e');
  if (!mday_conv_or)
    return -mday_conv_or.error();
  mday_conv = mday_conv_or.value();

  auto year_conv_or = get_specific_int_format(timeptr, to_conv, 'Y');
  if (!year_conv_or)
    return -year_conv_or.error();
  year_conv = year_conv_or.value();

  FormatSection raw_time_conv = to_conv;
  raw_time_conv.conv_name = 'T';
  raw_time_conv.min_width = 0;

  if (requested_padding > 0)
    RET_IF_RESULT_NEGATIVE(writer->write(' ', requested_padding));
  RET_IF_RESULT_NEGATIVE(writer->write(wday_str));
  RET_IF_RESULT_NEGATIVE(writer->write(' '));
  RET_IF_RESULT_NEGATIVE(writer->write(month_str));
  RET_IF_RESULT_NEGATIVE(writer->write(' '));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, mday_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(' '));
  RET_IF_RESULT_NEGATIVE(convert_time_second(writer, raw_time_conv, timeptr));
  RET_IF_RESULT_NEGATIVE(writer->write(' '));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, year_conv));

  return WRITE_OK;
}

template <printf_core::WriteMode write_mode>
LIBC_INLINE int convert_composite(printf_core::Writer<write_mode> *writer,
                                  const FormatSection &to_conv,
                                  const tm *timeptr) {
  switch (to_conv.conv_name) {
  case 'c': // locale specified date and time
            // in default locale Equivalent to %a %b %e %T %Y.
    return convert_full_date_time(writer, to_conv, timeptr);
  case 'D': // %m/%d/%y (month/day/year)
    return convert_date_us(writer, to_conv, timeptr);
  case 'F': // %Y-%m-%d (year-month-day)
    return convert_date_iso(writer, to_conv, timeptr);
  case 'r': // %I:%M:%S %p (hour:minute:second AM/PM)
    return convert_time_am_pm(writer, to_conv, timeptr);
  case 'R': // %H:%M (hour:minute)
    return convert_time_minute(writer, to_conv, timeptr);
  case 'T': // %H:%M:%S (hour:minute:second)
    return convert_time_second(writer, to_conv, timeptr);
  case 'x': // locale specified date
            // in default locale Equivalent to %m/%d/%y. (same as %D)
    return convert_date_us(writer, to_conv, timeptr);
  case 'X': // locale specified time
            // in default locale Equivalent to %T.
    return convert_time_second(writer, to_conv, timeptr);
  default:
    __builtin_trap(); // this should be unreachable, but trap if you hit it.
  }
}
} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_COMPOSITE_CONVERTER_H
