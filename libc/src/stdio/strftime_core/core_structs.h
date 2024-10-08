//===-- Core Structures for printf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CORE_STRUCTS_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CORE_STRUCTS_H

#include "src/__support/macros/config.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"

#include <inttypes.h>
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

struct tm {
	int	tm_sec;		/* seconds after the minute [0-60] */
	int	tm_min;		/* minutes after the hour [0-59] */
	int	tm_hour;	/* hours since midnight [0-23] */
	int	tm_mday;	/* day of the month [1-31] */
	int	tm_mon;		/* months since January [0-11] */
	int	tm_year;	/* years since 1900 */
	int	tm_wday;	/* days since Sunday [0-6] */
	int	tm_yday;	/* days since January 1 [0-365] */
	int	tm_isdst;	/* Daylight Savings Time flag */
	long	tm_gmtoff;	/* offset from UTC in seconds */
	char	*tm_zone;	/* timezone abbreviation */
};

struct FormatSection {
  bool has_conv {false};
  bool isE {false};
  bool isO {false};
  cpp::string_view raw_string;
  char conv_name;
  const struct tm *time;
};

enum PrimaryType : uint8_t {
  Unknown = 0,
  Float = 1,
  Pointer = 2,
  Integer = 3,
  FixedPoint = 4,
};

// TypeDesc stores the information about a type that is relevant to printf in
// a relatively compact manner.
struct TypeDesc {
  uint8_t size;
  PrimaryType primary_type;
  LIBC_INLINE constexpr bool operator==(const TypeDesc &other) const {
    return (size == other.size) && (primary_type == other.primary_type);
  }
};

template <typename T> LIBC_INLINE constexpr TypeDesc type_desc_from_type() {
  if constexpr (cpp::is_same_v<T, void>) {
    return TypeDesc{0, PrimaryType::Unknown};
  } else {
    constexpr bool IS_POINTER = cpp::is_pointer_v<T>;
    constexpr bool IS_FLOAT = cpp::is_floating_point_v<T>;
#ifdef LIBC_INTERNAL_STRFTIME_HAS_FIXED_POINT
    constexpr bool IS_FIXED_POINT = cpp::is_fixed_point_v<T>;
#else
    constexpr bool IS_FIXED_POINT = false;
#endif // LIBC_INTERNAL_STRFTIME_HAS_FIXED_POINT

    return TypeDesc{sizeof(T), IS_POINTER       ? PrimaryType::Pointer
                               : IS_FLOAT       ? PrimaryType::Float
                               : IS_FIXED_POINT ? PrimaryType::FixedPoint
                                                : PrimaryType::Integer};
  }
}

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
