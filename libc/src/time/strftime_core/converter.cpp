//===-- Format specifier converter implmentation for strftime -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See htto_conv.times://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/core_structs.h"

// #include "composite_converter.h"
#include "num_converter.h"
// #include "str_converter.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

int convert(printf_core::Writer *writer, const FormatSection &to_conv,
            const tm *timeptr) {
  // TODO: Implement the locale support.
  if (to_conv.modifier != ConvModifier::none) {
    return writer->write(to_conv.raw_string);
  }

  if (!to_conv.has_conv)
    return writer->write(to_conv.raw_string);
  switch (to_conv.conv_name) {
    // The cases are grouped by type, then alphabetized with lowercase before
    // uppercase.

    // raw conversions
  case '%':
    return writer->write("%");
  case 'n':
    return writer->write("\n");
  case 't':
    return writer->write("\t");

    // numeric conversions
  case 'C': // Century
  case 'd': // Day of the month [01-31]
  case 'e': // Day of the month [1-31]
  case 'g': // last 2 digits of ISO year
  case 'G': // ISO year
  case 'H': // 24-hour format
  case 'I': // 12-hour format
  case 'j': // Day of the year
  case 'm': // Month of the year
  case 'M': // Minute of the hour
  case 's': // Seconds since the epoch
  case 'S': // Second of the minute
  case 'u': // ISO day of the week ([1-7] starting Monday)
  case 'U': // Week of the year ([00-53] week 1 starts on first *Sunday*)
  case 'V': // ISO week number ([01-53], 01 is first week majority in this year)
  case 'w': // Day of week ([0-6] starting Sunday)
  case 'W': // Week of the year ([00-53] week 1 starts on first *Monday*)
  case 'y': // Year of the Century
  case 'Y': // Full year
    return convert_int(writer, to_conv, timeptr);

    // string conversions
  case 'a': // Abbreviated weekday name
  case 'A': // Full weekday name
  case 'b': // Abbreviated month name
  case 'B': // Full month name
  case 'h': // same as %b
  case 'p': // AM/PM designation
  case 'z': // Timezone offset (+/-hhmm)
  case 'Z': // Timezone name
            // return write_str(writer, to_conv, timeptr);

    // composite conversions
  case 'c': // locale specified date and time
  case 'D': // %m/%d/%y (month/day/year)
  case 'F': // %Y-%m-%d (year-month-day)
  case 'r': // %I:%M:%S %p (hour:minute:second AM/PM)
  case 'R': // %H:%M (hour:minute)
  case 'T': // %H:%M:%S (hour:minute:second)
  case 'x': // locale specified date
  case 'X': // locale specified time
            // return write_composite(writer, to_conv, timeptr);
  default:
    return writer->write(to_conv.raw_string);
  }
  return 0;
}

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL
