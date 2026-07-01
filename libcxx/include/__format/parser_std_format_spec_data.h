// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_PARSER_STD_FORMAT_SPEC_DATA_H
#define _LIBCPP___FORMAT_PARSER_STD_FORMAT_SPEC_DATA_H

#include <__config>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __format_spec {

enum class __alignment : uint8_t {
  /// No alignment is set in the format string.
  __default,
  __left,
  __center,
  __right,
  __zero_padding
};

enum class __sign : uint8_t {
  /// No sign is set in the format string.
  ///
  /// The sign isn't allowed for certain format-types. By using this value
  /// it's possible to detect whether or not the user explicitly set the sign
  /// flag. For formatting purposes it behaves the same as \ref __minus.
  __default,
  __minus,
  __plus,
  __space
};

enum class __type : uint8_t {
  __default = 0,
  __string,
  __binary_lower_case,
  __binary_upper_case,
  __octal,
  __decimal,
  __hexadecimal_lower_case,
  __hexadecimal_upper_case,
  __pointer_lower_case,
  __pointer_upper_case,
  __char,
  __hexfloat_lower_case,
  __hexfloat_upper_case,
  __scientific_lower_case,
  __scientific_upper_case,
  __fixed_lower_case,
  __fixed_upper_case,
  __general_lower_case,
  __general_upper_case,
  __debug
};

// The fill UCS scalar value.
//
// This is always an array, with 1, 2, or 4 elements.
// The size of the data structure is always 32-bits.
template <class _CharT>
struct __code_point;

template <>
struct __code_point<char> {
  char __data[4] = {' '};
};

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <>
struct __code_point<wchar_t> {
  wchar_t __data[4 / sizeof(wchar_t)] = {L' '};
};
#  endif

template <class _CharT>
struct __parser_data {
  __alignment __alignment_     : 3 {__alignment::__default};
  __sign __sign_               : 2 {__sign::__default};
  bool __alternate_form_       : 1 {false};
  bool __locale_specific_form_ : 1 {false};
  bool __clear_brackets_       : 1 {false};
  __type __type_{__type::__default};

  // These flags are only used for formatting chrono. Since the struct has
  // padding space left it's added to this structure.
  bool __hour_ : 1 {false};

  bool __weekday_name_ : 1 {false};
  bool __weekday_      : 1 {false};

  bool __day_of_year_  : 1 {false};
  bool __week_of_year_ : 1 {false};

  bool __month_name_ : 1 {false};

  uint8_t __reserved_0_ : 2 {0};
  uint8_t __reserved_1_ : 6 {0};
  // These two flags are only used internally and not part of the
  // __parsed_specifications. Therefore put them at the end.
  bool __width_as_arg_     : 1 {false};
  bool __precision_as_arg_ : 1 {false};

  /// The requested width, either the value or the arg-id.
  int32_t __width_{0};

  /// The requested precision, either the value or the arg-id.
  int32_t __precision_{-1};

  __code_point<_CharT> __fill_{};
};

} // namespace __format_spec

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___FORMAT_PARSER_STD_FORMAT_SPEC_DATA_H
