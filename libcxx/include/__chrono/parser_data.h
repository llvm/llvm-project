// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_PARSER_DATA_H
#define _LIBCPP___CHRONO_PARSER_DATA_H

#include <__config>

#if _LIBCPP_HAS_LOCALIZATION

#  include <cstdint>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

#  if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

namespace __parser {

// Fields supplied by the parsed format directives.
enum class __fields_set : uint32_t {
  __none    = 0,
  __year    = 1 << 0,
  __month   = 1 << 1,
  __day     = 1 << 2,
  __hours   = 1 << 3,
  __minutes = 1 << 4,
  __seconds = 1 << 5,

  __iso_year = 1 << 6,
  __iso_week = 1 << 7,
  __weekday  = 1 << 8,

  __century         = 1 << 9,
  __year_of_century = 1 << 10,

  __hour12 = 1 << 11,
  __am_pm  = 1 << 12,

  __day_of_year = 1 << 13,
  __week_sun    = 1 << 14,
  __week_mon    = 1 << 15,

  __utc_offset = 1 << 16,

  __iso_year_of_century = 1 << 17,
};

_LIBCPP_HIDE_FROM_ABI constexpr __fields_set operator|(__fields_set __lhs, __fields_set __rhs) {
  return static_cast<__fields_set>(static_cast<uint32_t>(__lhs) | static_cast<uint32_t>(__rhs));
}

_LIBCPP_HIDE_FROM_ABI constexpr __fields_set operator&(__fields_set __lhs, __fields_set __rhs) {
  return static_cast<__fields_set>(static_cast<uint32_t>(__lhs) & static_cast<uint32_t>(__rhs));
}

_LIBCPP_HIDE_FROM_ABI constexpr __fields_set operator~(__fields_set __fields) {
  return static_cast<__fields_set>(~static_cast<uint32_t>(__fields));
}

_LIBCPP_HIDE_FROM_ABI constexpr __fields_set& operator|=(__fields_set& __lhs, __fields_set __rhs) {
  return __lhs = __lhs | __rhs;
}

} // namespace __parser

using __parser::__fields_set;

// Fields collected while parsing. Date/time construction and validation do not
// modify these values or their presence flags.
struct __fields_storage {
  int __year_  = 0;
  int __month_ = 0;
  int __day_   = 0;
  // Time-of-day fields.
  int __hours_   = 0;
  int __minutes_ = 0;
  int __seconds_ = 0;
  // Fractional seconds are stored as an integer number of attoseconds.
  static constexpr unsigned __subsecond_precision = 18;
  int64_t __subseconds_                           = 0;

  // __weekday_ uses [0, 6], with Sunday == 0.
  int __iso_year_            = 0;
  int __iso_year_of_century_ = 0;
  int __iso_week_            = 0;
  int __weekday_             = 0;

  int __century_         = 0;
  int __year_of_century_ = 0;

  int __hour12_ = 0;
  bool __is_pm_ = false;

  // For durations, __day_of_year_ is a number of days and is not limited to [1, 366].
  int __day_of_year_ = 0;
  int __week_sun_    = 0;
  int __week_mon_    = 0;

  // Minutes east of UTC.
  int __utc_offset_ = 0;

  __fields_set __present_ = __fields_set::__none;

  _LIBCPP_HIDE_FROM_ABI constexpr void __set(__fields_set __field_bit) { __present_ |= __field_bit; }

  // Check whether all specified fields are present, allowing additional fields.
  // __utc_offset is checked like any other field.
  _LIBCPP_HIDE_FROM_ABI constexpr bool __has(__fields_set __parts) const { return (__present_ & __parts) == __parts; }

  // Check whether at least one specified field is present, allowing additional fields.
  // __utc_offset is checked like any other field.
  _LIBCPP_HIDE_FROM_ABI constexpr bool __has_any(__fields_set __parts) const {
    return (__present_ & __parts) != __fields_set::__none;
  }

  // Allow only these fields, without requiring all of them to be present.
  // Ignore __utc_offset so callers do not have to explicitly allow it each time.
  _LIBCPP_HIDE_FROM_ABI constexpr bool __has_only(__fields_set __allowed) const {
    const auto __fields = __present_ & ~__fields_set::__utc_offset;
    return (__fields & ~__allowed) == __fields_set::__none;
  }

  // Require every required field and allow only optional fields in addition.
  // The two sets must be disjoint. Exclude the UTC offset from this check
  // so callers do not have to explicitly allow __utc_offset each time.
  _LIBCPP_HIDE_FROM_ABI constexpr bool
  __has_exactly(__fields_set __required, __fields_set __optional = __fields_set::__none) const {
    const auto __fields = __present_ & ~__fields_set::__utc_offset;
    return (__fields & ~__optional) == __required;
  }
};

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD

#  endif
#endif

#endif //_LIBCPP___CHRONO_PARSER_DATA_H
