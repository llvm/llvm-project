// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_FORMATTER_BOOL_H
#define _LIBCPP___FORMAT_FORMATTER_BOOL_H

#include <__config>
#include <__format/concepts.h>
#include <__format/formatter.h>
#include <__format/parser_std_format_spec.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

template <__fmt_char_type _CharT, class _FormatContext>
_LIBCPP_HIDE_FROM_ABI typename _FormatContext::iterator
__formatter_bool_format(bool __value, __format_spec::__parser<_CharT>, _FormatContext&);

template <__fmt_char_type _CharT>
struct formatter<bool, _CharT> {
  template <class _ParseContext>
  _LIBCPP_HIDE_FROM_ABI constexpr typename _ParseContext::iterator parse(_ParseContext& __ctx) {
    typename _ParseContext::iterator __result = __parser_.__parse(__ctx, __format_spec::__fields_integral);
    __format_spec::__process_parsed_bool(__parser_, "a bool");
    return __result;
  }

  template <class _FormatContext>
  _LIBCPP_HIDE_FROM_ABI typename _FormatContext::iterator format(bool __value, _FormatContext& __ctx) const {
    return std::__formatter_bool_format(__value, __parser_, __ctx);
  }

  __format_spec::__parser<_CharT> __parser_;
};

#  if _LIBCPP_STD_VER >= 23
template <>
inline constexpr bool enable_nonlocking_formatter_optimization<bool> = true;
#  endif // _LIBCPP_STD_VER >= 23
#endif   // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FORMAT_FORMATTER_BOOL_H
