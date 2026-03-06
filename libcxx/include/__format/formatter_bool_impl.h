// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_FORMATTER_BOOL_IMPL_H
#define _LIBCPP___FORMAT_FORMATTER_BOOL_IMPL_H

#include <__assert>
#include <__config>
#include <__format/concepts.h>
#include <__format/formatter_bool.h>
#include <__format/formatter_integral.h>
#include <__format/parser_std_format_spec.h>
#include <__utility/unreachable.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

template <__fmt_char_type _CharT, class _FormatContext>
_LIBCPP_HIDE_FROM_ABI typename _FormatContext::iterator
__formatter_bool_format(bool __value, __format_spec::__parser<_CharT> __parser, _FormatContext& __ctx) {
  switch (__parser.__type_) {
  case __format_spec::__type::__default:
  case __format_spec::__type::__string:
    return __formatter::__format_bool(__value, __ctx, __parser.__get_parsed_std_specifications(__ctx));

  case __format_spec::__type::__binary_lower_case:
  case __format_spec::__type::__binary_upper_case:
  case __format_spec::__type::__octal:
  case __format_spec::__type::__decimal:
  case __format_spec::__type::__hexadecimal_lower_case:
  case __format_spec::__type::__hexadecimal_upper_case:
    // Promotes bool to an integral type. This reduces the number of
    // instantiations of __format_integer reducing code size.
    return __formatter::__format_integer(
        static_cast<unsigned>(__value), __ctx, __parser.__get_parsed_std_specifications(__ctx));

  default:
    _LIBCPP_ASSERT_INTERNAL(false, "The parse function should have validated the type");
    __libcpp_unreachable();
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___FORMAT_FORMATTER_BOOL_IMPL_H
