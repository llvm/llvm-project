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
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

using DecFmt = IntegerToString<uintmax_t>;

LIBC_INLINE int convert_int(printf_core::Writer *writer,
                            const FormatSection &to_conv, const tm *timeptr) {
  const time_utils::TMReader time_reader(timeptr);

  intmax_t raw_num;
  size_t pad_to_len;

  // gets_plus_sign is only true for year conversions where the year would be
  // positive and more than 4 digits, including leading spaces. Both the
  // FORCE_SIGN flag and gets_plus_sign must be true for a plus sign to be
  // output.
  bool gets_plus_sign = false;

  switch (to_conv.conv_name) {
  case 'C': // Century [00-99]
    raw_num = time_reader.get_year() / 100;
    gets_plus_sign = raw_num > 99 || to_conv.min_width > 2;
    pad_to_len = 2;
    break;
  case 'd': // Day of the month [01-31]
    raw_num = time_reader.get_mday();
    pad_to_len = 2;
    break;
  case 'e': // Day of the month [1-31]
    raw_num = time_reader.get_mday();
    pad_to_len = 1;
    break;
  case 'g': // last 2 digits of ISO year [00-99]
    raw_num = time_reader.get_iso_year() % 100;
    pad_to_len = 2;
    break;
  case 'G': // ISO year
    raw_num = time_reader.get_iso_year();
    gets_plus_sign = raw_num > 9999 || to_conv.min_width > 4;
    pad_to_len = 0;
    break;
  case 'H': // 24-hour format [00-23]
    raw_num = time_reader.get_hour();
    pad_to_len = 2;
    break;
  case 'I': // 12-hour format [01-12]
    raw_num = ((time_reader.get_hour() + 11) % 12) + 1;
    pad_to_len = 2;
    break;
  case 'j':                               // Day of the year [001-366]
    raw_num = time_reader.get_yday() + 1; // get_yday is 0 indexed
    pad_to_len = 3;
    break;
  case 'm':                              // Month of the year [01-12]
    raw_num = time_reader.get_mon() + 1; // get_mon is 0 indexed
    pad_to_len = 2;
    break;
  case 'M': // Minute of the hour [00-59]
    raw_num = time_reader.get_min();
    pad_to_len = 2;
    break;
  case 's': // Seconds since the epoch
    raw_num = time_reader.get_epoch();
    pad_to_len = 0;
    break;
  case 'S': // Second of the minute [00-60]
    raw_num = time_reader.get_sec();
    pad_to_len = 2;
    break;
  case 'u': // ISO day of the week ([1-7] starting Monday)
    raw_num = time_reader.get_iso_wday() + 1;
    // need to add 1 because get_iso_wday returns the weekday [0-6].
    pad_to_len = 1;
    break;
  case 'U': // Week of the year ([00-53] week 1 starts on first *Sunday*)
    // This doesn't actually end up using tm_year, despite the standard saying
    // it's needed. The end of the current year doesn't really matter, so leap
    // years aren't relevant. If this is wrong, please tell me what I'm missing.
    raw_num = time_reader.get_week(time_constants::SUNDAY);
    pad_to_len = 2;
    break;
  case 'V': // ISO week number ([01-53], 01 is first week majority in this year)
    // This does need to know the year, since it may affect what the week of the
    // previous year it underflows to.
    raw_num = time_reader.get_iso_week();
    pad_to_len = 2;
    break;
  case 'w': // Day of week ([0-6] starting Sunday)
    raw_num = time_reader.get_wday();
    pad_to_len = 1;
    break;
  case 'W': // Week of the year ([00-53] week 1 starts on first *Monday*)
    raw_num = time_reader.get_week(time_constants::MONDAY);
    pad_to_len = 2;
    break;
  case 'y': // Year of the Century [00-99]
    raw_num = time_reader.get_year() % 100;
    pad_to_len = 2;
    break;
  case 'Y': // Full year
    raw_num = time_reader.get_year();
    gets_plus_sign = raw_num > 9999 || to_conv.min_width > 4;
    pad_to_len = 0;
    break;
  default:
    __builtin_trap(); // this should be unreachable, but trap if you hit it.
  }

  const uintmax_t num =
      static_cast<uintmax_t>(raw_num < 0 ? -raw_num : raw_num);
  const bool is_negative = raw_num < 0;

  DecFmt d(num);
  auto str = d.view();

  size_t digits_written = str.size();

  char sign_char = 0;

  // TODO: Handle locale modifiers

  if (is_negative)
    sign_char = '-';
  else if ((to_conv.flags & FormatFlags::FORCE_SIGN) ==
               FormatFlags::FORCE_SIGN &&
           gets_plus_sign)
    sign_char = '+';

  // sign isn't a problem because we're taking the max. The result is always
  // non-negative.
  if (to_conv.min_width > 0)
    pad_to_len = to_conv.min_width;

  // one less digit of padding if there's a sign char
  int zeroes =
      static_cast<int>(pad_to_len - digits_written - (sign_char == 0 ? 0 : 1));

  // Format is (sign) (zeroes) digits
  if (sign_char != 0)
    RET_IF_RESULT_NEGATIVE(writer->write(sign_char));
  if (zeroes > 0)
    RET_IF_RESULT_NEGATIVE(writer->write('0', zeroes))
  RET_IF_RESULT_NEGATIVE(writer->write(str));

  return WRITE_OK;
}

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif
