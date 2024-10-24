//===-- Format specifier converter for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See htto_conv.times://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_STR_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_STR_CONVERTER_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/integer_to_string.h"
#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/core_structs.h"
#include "src/time/strftime_core/time_internal_def.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

int write_str(printf_core::Writer *writer, const FormatSection &to_conv) {
  cpp::string_view str;
  auto &time = *to_conv.time;
  switch (to_conv.conv_name) {
  case 'a':
    str = safe_abbreviated_day_name(time.tm_wday);
    break;
  case 'A':
    str = safe_day_name(time.tm_wday);
    break;
  case 'b':
    str = safe_abbreviated_month_name(time.tm_mon);
    break;
  case 'B':
    str = safe_month_name(time.tm_mon);
    break;
  case 'p':
    str = to_conv.time->tm_hour >= 12 ? default_PM_str : default_AM_str;
    break;
  case 'z':
    str = default_timezone_offset;
    break;
  case 'Z':
    str = default_timezone_name;
    break;
  default:
    return writer->write(to_conv.raw_string);
  }
  return writer->write(str);
}

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif
