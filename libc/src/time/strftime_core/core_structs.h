//===-- Core Structures for printf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CORE_STRUCTS_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CORE_STRUCTS_H

#include "src/__support/CPP/string_view.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

struct tm {
  int tm_sec;     /* seconds after the minute [0-60] */
  int tm_min;     /* minutes after the hour [0-59] */
  int tm_hour;    /* hours since midnight [0-23] */
  int tm_mday;    /* day of the month [1-31] */
  int tm_mon;     /* months since January [0-11] */
  int tm_year;    /* years since 1900 */
  int tm_wday;    /* days since Sunday [0-6] */
  int tm_yday;    /* days since January 1 [0-365] */
  int tm_isdst;   /* Daylight Savings Time flag */
  long tm_gmtoff; /* offset from UTC in seconds */
  char *tm_zone;  /* timezone abbreviation */
};

struct FormatSection {
  bool has_conv{false};
  bool isE{false};
  bool isO{false};
  cpp::string_view raw_string{};
  char conv_name;
  const struct tm *time;

  // This operator is only used for testing and should be automatically
  // optimized out for release builds.
  LIBC_INLINE bool operator==(const FormatSection &other) const {
    if (has_conv != other.has_conv)
      return false;
    if (raw_string != other.raw_string)
      return false;
    return true;
  }
};

// This is the value to be returned by conversions when no error has occurred.
constexpr int WRITE_OK = 0;
// These are the printf return values for when an error has occurred. They are
// all negative, and should be distinct.
constexpr int FILE_WRITE_ERROR = -1;
constexpr int FILE_STATUS_ERROR = -2;
constexpr int NULLPTR_WRITE_ERROR = -3;
constexpr int INT_CONVERSION_ERROR = -4;
constexpr int FIXED_POINT_CONVERSION_ERROR = -5;
constexpr int ALLOCATION_ERROR = -6;
} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CORE_STRUCTS_H
