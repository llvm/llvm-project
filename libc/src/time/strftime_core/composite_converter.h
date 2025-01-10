//===-- Composite converter for strftime ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See htto_conv.times://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_COMPOSITE_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_COMPOSITE_CONVERTER_H

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

LIBC_INLINE int convert_date_us(printf_core::Writer *writer,
                                const FormatSection &to_conv,
                                const tm *timeptr) {
  // format is %m/%d/%y (month/day/year)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  const size_t trailing_conv_len = 1 + 2 + 1 + 2; // sizeof("/01/02")
  IntFormatSection year_conv;
  IntFormatSection mon_conv;
  IntFormatSection mday_conv;

  {
    FormatSection raw_mon_conv = to_conv;
    raw_mon_conv.conv_name = 'm';

    const int requested_padding = to_conv.min_width - trailing_conv_len;
    // a negative padding will be treated as the default
    raw_mon_conv.min_width = requested_padding;

    mon_conv = get_int_format(raw_mon_conv, timeptr);

    // If the user set the padding, but it's below the width of the trailing
    // conversions, then there should be no padding.
    if (to_conv.min_width > 0 && requested_padding < 0)
      mon_conv.pad_to_len = 0;
  }
  {
    FormatSection raw_mday_conv = to_conv;
    raw_mday_conv.conv_name = 'd';
    raw_mday_conv.min_width = 0;

    mday_conv = get_int_format(raw_mday_conv, timeptr);
  }
  {
    FormatSection raw_year_conv = to_conv;
    raw_year_conv.conv_name = 'y';
    raw_year_conv.min_width = 0;

    year_conv = get_int_format(raw_year_conv, timeptr);
  }

  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, mon_conv));
  RET_IF_RESULT_NEGATIVE(writer->write('/'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, mday_conv));
  RET_IF_RESULT_NEGATIVE(writer->write('/'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, year_conv));

  return WRITE_OK;
}

LIBC_INLINE int convert_date_iso(printf_core::Writer *writer,
                                 const FormatSection &to_conv,
                                 const tm *timeptr) {
  // format is "%Y-%m-%d" (year-month-day)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  const size_t trailing_conv_len = 1 + 2 + 1 + 2; // sizeof("-01-02")
  IntFormatSection year_conv;
  IntFormatSection mon_conv;
  IntFormatSection mday_conv;

  {
    FormatSection raw_year_conv = to_conv;
    raw_year_conv.conv_name = 'Y';

    const int requested_padding = to_conv.min_width - trailing_conv_len;
    // a negative padding will be treated as the default
    raw_year_conv.min_width = requested_padding;

    year_conv = get_int_format(raw_year_conv, timeptr);

    // If the user set the padding, but it's below the width of the trailing
    // conversions, then there should be no padding.
    if (to_conv.min_width > 0 && requested_padding < 0)
      year_conv.pad_to_len = 0;
  }
  {
    FormatSection raw_mon_conv = to_conv;
    raw_mon_conv.conv_name = 'm';
    raw_mon_conv.min_width = 0;

    mon_conv = get_int_format(raw_mon_conv, timeptr);
  }
  {
    FormatSection raw_mday_conv = to_conv;
    raw_mday_conv.conv_name = 'd';
    raw_mday_conv.min_width = 0;

    mday_conv = get_int_format(raw_mday_conv, timeptr);
  }

  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, year_conv));
  RET_IF_RESULT_NEGATIVE(writer->write('-'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, mon_conv));
  RET_IF_RESULT_NEGATIVE(writer->write('-'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, mday_conv));

  return WRITE_OK;
}

LIBC_INLINE int convert_time_am_pm(printf_core::Writer *writer,
                                   const FormatSection &to_conv,
                                   const tm *timeptr) {
  // format is "%I:%M:%S %p" (hour:minute:second AM/PM)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  const size_t trailing_conv_len = 1 + 2 + 1 + 2 + 1 + 2; // sizeof(":01:02 AM")
  IntFormatSection hour_conv;
  IntFormatSection min_conv;
  IntFormatSection sec_conv;

  const time_utils::TMReader time_reader(timeptr);

  {
    FormatSection raw_hour_conv = to_conv;
    raw_hour_conv.conv_name = 'I';

    const int requested_padding = to_conv.min_width - trailing_conv_len;
    // a negative padding will be treated as the default
    raw_hour_conv.min_width = requested_padding;

    hour_conv = get_int_format(raw_hour_conv, timeptr);

    // If the user set the padding, but it's below the width of the trailing
    // conversions, then there should be no padding.
    if (to_conv.min_width > 0 && requested_padding < 0)
      hour_conv.pad_to_len = 0;
  }
  {
    FormatSection raw_min_conv = to_conv;
    raw_min_conv.conv_name = 'M';
    raw_min_conv.min_width = 0;

    min_conv = get_int_format(raw_min_conv, timeptr);
  }
  {
    FormatSection raw_sec_conv = to_conv;
    raw_sec_conv.conv_name = 'S';
    raw_sec_conv.min_width = 0;

    sec_conv = get_int_format(raw_sec_conv, timeptr);
  }

  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, hour_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, min_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, sec_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(' '));
  RET_IF_RESULT_NEGATIVE(writer->write(time_reader.get_am_pm()));

  return WRITE_OK;
}

LIBC_INLINE int convert_time_minute(printf_core::Writer *writer,
                                    const FormatSection &to_conv,
                                    const tm *timeptr) {
  // format is "%H:%M" (hour:minute)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  const size_t trailing_conv_len = 1 + 2; // sizeof(":01")
  IntFormatSection hour_conv;
  IntFormatSection min_conv;

  {
    FormatSection raw_hour_conv = to_conv;
    raw_hour_conv.conv_name = 'H';

    const int requested_padding = to_conv.min_width - trailing_conv_len;
    // a negative padding will be treated as the default
    raw_hour_conv.min_width = requested_padding;

    hour_conv = get_int_format(raw_hour_conv, timeptr);

    // If the user set the padding, but it's below the width of the trailing
    // conversions, then there should be no padding.
    if (to_conv.min_width > 0 && requested_padding < 0)
      hour_conv.pad_to_len = 0;
  }
  {
    FormatSection raw_min_conv = to_conv;
    raw_min_conv.conv_name = 'M';
    raw_min_conv.min_width = 0;

    min_conv = get_int_format(raw_min_conv, timeptr);
  }
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, hour_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, min_conv));

  return WRITE_OK;
}

LIBC_INLINE int convert_time_second(printf_core::Writer *writer,
                                    const FormatSection &to_conv,
                                    const tm *timeptr) {
  // format is "%H:%M:%S" (hour:minute:second)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  const size_t trailing_conv_len = 1 + 2 + 1 + 2; // sizeof(":01:02")
  IntFormatSection hour_conv;
  IntFormatSection min_conv;
  IntFormatSection sec_conv;

  {
    FormatSection raw_hour_conv = to_conv;
    raw_hour_conv.conv_name = 'H';

    const int requested_padding = to_conv.min_width - trailing_conv_len;
    // a negative padding will be treated as the default
    raw_hour_conv.min_width = requested_padding;

    hour_conv = get_int_format(raw_hour_conv, timeptr);

    // If the user set the padding, but it's below the width of the trailing
    // conversions, then there should be no padding.
    if (to_conv.min_width > 0 && requested_padding < 0)
      hour_conv.pad_to_len = 0;
  }
  {
    FormatSection raw_min_conv = to_conv;
    raw_min_conv.conv_name = 'M';
    raw_min_conv.min_width = 0;

    min_conv = get_int_format(raw_min_conv, timeptr);
  }
  {
    FormatSection raw_sec_conv = to_conv;
    raw_sec_conv.conv_name = 'S';
    raw_sec_conv.min_width = 0;

    sec_conv = get_int_format(raw_sec_conv, timeptr);
  }

  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, hour_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, min_conv));
  RET_IF_RESULT_NEGATIVE(writer->write(':'));
  RET_IF_RESULT_NEGATIVE(write_padded_int(writer, sec_conv));

  return WRITE_OK;
}

LIBC_INLINE int convert_full_date_time(printf_core::Writer *writer,
                                       const FormatSection &to_conv,
                                       const tm *timeptr) {
  const time_utils::TMReader time_reader(timeptr);
  // format is "%a %b %e %T %Y" (weekday month mday [time] year)
  // we only pad the first conversion, and we assume all the other values are in
  // their valid ranges.
  // sizeof("Sun Jan 12 03:45:06 2025")
  const size_t full_conv_len = 3 + 1 + 3 + 1 + 2 + 1 + 8 + 1 + 4;
  // use the full conv lent because
  const int requested_padding = to_conv.min_width - full_conv_len;

  cpp::string_view wday_str = unwrap_opt(time_reader.get_weekday_short_name());
  cpp::string_view month_str = unwrap_opt(time_reader.get_month_short_name());
  IntFormatSection mday_conv;
  FormatSection raw_time_conv = to_conv;
  IntFormatSection year_conv;

  {
    FormatSection raw_mday_conv = to_conv;
    raw_mday_conv.conv_name = 'e';
    raw_mday_conv.min_width = 0;

    mday_conv = get_int_format(raw_mday_conv, timeptr);
  }
  {
    FormatSection raw_year_conv = to_conv;
    raw_year_conv.conv_name = 'Y';
    raw_year_conv.min_width = 0;

    year_conv = get_int_format(raw_year_conv, timeptr);
  }
  {
    raw_time_conv.conv_name = 'T';
    raw_time_conv.min_width = 0;
  }

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

LIBC_INLINE int convert_composite(printf_core::Writer *writer,
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
