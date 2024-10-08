//===-- Format specifier converter for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CONVERTER_H

#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/config.h"
#include "src/math/log10.h"
#include "src/stdio/printf_core/writer.h"
#include "src/stdio/strftime_core/core_structs.h"
#include "src/stdio/strftime_core/time_internal_def.h"
#include <time.h>

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

namespace details {

LIBC_INLINE cpp::optional<cpp::string_view>
num_to_strview(uintmax_t num, cpp::span<char> bufref) {
  return IntegerToString<uintmax_t>::format_to(bufref, num);
}

template<int width>
LIBC_INLINE int write_num(uintmax_t num,
                                       printf_core::Writer *writer) {
  cpp::array<char, width> buf;
  return writer->write(*num_to_strview(num, buf));
}

template <int width, char padding>
LIBC_INLINE int write_num_with_padding(uintmax_t num,
                                       printf_core::Writer *writer) {
  cpp::array<char, width> buf;
  auto digits = log10(num) + 1;
  auto padding_needed = width - digits;
 int char_written = 0;
  for (int _ = 0; _ < padding_needed; _++) {
    char_written += writer->write(padding);
  }
  char_written += writer->write(*num_to_strview(num, buf));
  return char_written;
}

} // namespace details

LIBC_INLINE int convert_weekday(printf_core::Writer *writer,
                           const FormatSection &to_conv) {
  return writer->write(day_names[to_conv.time->tm_wday]);
}

LIBC_INLINE int convert_zero_padded_day_of_year(printf_core::Writer *writer,
                                            const FormatSection &to_conv) {
  return details::write_num_with_padding<3, '0'>(to_conv.time->tm_yday + 1, writer);
}

LIBC_INLINE int convert_zero_padded_day_of_month(printf_core::Writer *writer,
                                            const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>(to_conv.time->tm_mday, writer);
}

LIBC_INLINE int convert_space_padded_day_of_month(printf_core::Writer *writer,
                                             const FormatSection &to_conv) {
  return details::write_num_with_padding<2, ' '>(to_conv.time->tm_mday, writer);
}

LIBC_INLINE int convert_decimal_weekday(printf_core::Writer *writer,
                                             const FormatSection &to_conv) {
  return details::write_num<1>(to_conv.time->tm_wday == 0 ? 7 : to_conv.time->tm_wday, writer);
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
  return writer->write(month_names[to_conv.time->tm_mon]);
}

LIBC_INLINE int convert_abbreviated_month(printf_core::Writer *writer,
                                          const FormatSection &to_conv) {
  return writer->write(abbreviated_month_names[to_conv.time->tm_mon]);
}

LIBC_INLINE int convert_zero_padded_month(printf_core::Writer *writer,
                                          const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>(to_conv.time->tm_mon + 1, writer);
}

LIBC_INLINE int convert_full_year(printf_core::Writer *writer,
                                  const FormatSection &to_conv) {
  return details::write_num_with_padding<4, '0'>(to_conv.time->tm_year + 1900, writer);
}

LIBC_INLINE int convert_two_digit_year(printf_core::Writer *writer,
                                       const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>((to_conv.time->tm_year + 1900) % 100, writer);
}

LIBC_INLINE int convert_century(printf_core::Writer *writer,
                                const FormatSection &to_conv) {
  return details::write_num_with_padding<2, '0'>((to_conv.time->tm_year + 1900) / 100, writer);
}

LIBC_INLINE int convert_iso_year(printf_core::Writer *writer,
                                 const FormatSection &to_conv) {
  // TODO
  return details::write_num_with_padding<4, '0'>(to_conv.time->tm_year + 1900, writer);
}

int convert(printf_core::Writer *writer, const FormatSection &to_conv) {
  if (!to_conv.has_conv)
    return writer->write(to_conv.raw_string);
  switch (to_conv.conv_name) {
  // day of the week
  case 'a':
    return convert_weekday(writer, to_conv);
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
  case 'V': // TODO: ISO 8061
  // month
  case 'B':
    return convert_full_month(writer, to_conv);
  case 'b':
  case 'h':
    return convert_abbreviated_month(writer, to_conv);
  case 'm':
    return convert_zero_padded_month(writer, to_conv);
  // year
  case 'Y':  // Full year (e.g., 2024)
    return convert_full_year(writer, to_conv);
  case 'y':  // Two-digit year (e.g., 24 for 2024)
    return convert_two_digit_year(writer, to_conv);
  case 'C':  // Century (e.g., 20 for 2024)
    return convert_century(writer, to_conv);
  case 'G':  // ISO 8601 year
    // TODO
    return convert_iso_year(writer, to_conv);

  }
  return 0;
}

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL
#endif
