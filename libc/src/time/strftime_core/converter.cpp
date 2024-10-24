//===-- Format specifier converter for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See htto_conv.times://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CONVERTER_H

#include "composite_converter.h"
#include "num_converter.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/core_structs.h"
#include "src/time/strftime_core/time_internal_def.h"
#include "str_converter.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

int convert(printf_core::Writer *writer, const FormatSection &to_conv) {
  if (!to_conv.has_conv)
    return writer->write(to_conv.raw_string);
  switch (to_conv.conv_name) {
  case '%':
    return writer->write("%");
  case 'C': // Century (C)
  case 'Y': // Full year (Y)
  case 'y': // Two-digit year (y)
  case 'j': // Day of the year (j)
  case 'm': // Month (m)
  case 'd': // Day of the month (d)
  case 'e': // Day of the month (e)
  case 'H': // 24-hour format (H)
  case 'I': // 12-hour format (I)
  case 'M': // Minute (M)
  case 'S': // Second (S)
  case 'U': // Week number starting on Sunday (U)
  case 'W': // Week number starting on Monday (W)
  case 'V': // ISO week number (V)
  case 'G': // ISO year (G)
  case 'w': // Decimal weekday (w)
  case 'u': // ISO weekday (u)
    return write_num(writer, to_conv);
  case 'a': // Abbreviated weekday name (a)
  case 'A': // Full weekday name (A)
  case 'b': // Abbreviated month name (b)
  case 'B': // Full month name (B)
  case 'p': // AM/PM designation (p)
  case 'z': // Timezone offset (z)
  case 'Z': // Timezone name (Z)
    return write_str(writer, to_conv);
  case 'c':
  case 'F':
  case 'r':
  case 'R':
  case 'T':
  case 'x':
  case 'X':
    return write_composite(writer, to_conv);
  default:
    return writer->write(to_conv.raw_string);
  }
  return 0;
}

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL
#endif
