// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___THREAD_FORMATTER_H
#define _LIBCPP___THREAD_FORMATTER_H

#include <__concepts/arithmetic.h>
#include <__config>
#include <__format/concepts.h>
#include <__format/format_parse_context.h>
#include <__format/formatter.h>
#include <__format/formatter_integral.h>
#include <__format/parser_std_format_spec.h>
#include <__thread/id.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_HAS_THREADS

template <__fmt_char_type _CharT>
struct formatter<__thread_id, _CharT> {
public:
  template <class _ParseContext>
  _LIBCPP_HIDE_FROM_ABI constexpr typename _ParseContext::iterator parse(_ParseContext& __ctx) {
    return __parser_.__parse(__ctx, __format_spec::__fields_fill_align_width);
  }

  template <class _FormatContext>
  _LIBCPP_HIDE_FROM_ABI typename _FormatContext::iterator format(__thread_id __id, _FormatContext& __ctx) const {
    __format_spec::__parsed_specifications<_CharT> __specs = __parser_.__get_parsed_std_specifications(__ctx);
    if constexpr (__thread_id::__PRINT_AS_HEX) {
      __specs.__std_.__alternate_form_ = true;
      __specs.__std_.__type_           = __format_spec::__type::__hexadecimal_lower_case;
    }
    return __formatter::__format_integer(__id.__get_formatter_value(), __ctx, __specs);
  }

  __format_spec::__parser<_CharT> __parser_{.__alignment_ = __format_spec::__alignment::__right};
};

#  endif // _LIBCPP_HAS_THREADS

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

#endif // _LIBCPP___THREAD_FORMATTER_H
