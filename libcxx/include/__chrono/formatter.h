// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_FORMATTER_H
#define _LIBCPP___CHRONO_FORMATTER_H

#include <__chrono/convert_to_tm.h>
#include <__chrono/day.h>
#include <__chrono/parser_std_format_spec.h>
#include <__config>
#include <__format/concepts.h>
#include <__format/format_parse_context.h>
#include <__format/formatter.h>
#include <__format/formatter_output.h>
#include <__format/parser_std_format_spec.h>
#include <ctime>
#include <sstream>
#include <string>
#include <string_view>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_FORMAT)

namespace __formatter {

/// Formats a time based on a tm struct.
///
/// This formatter passes the formatting to time_put which uses strftime. When
/// the value is outside the valid range it's unspecified what strftime will
/// output. For example weekday 8 can print 1 when the day is processed modulo
/// 7 since that handles the Sunday for 0-based weekday. It can also print 8 if
/// 7 is handled as a special case.
///
/// The Standard doesn't specify what to do in this case so the result depends
/// on the result of the underlying code.
///
/// \pre When the (abbreviated) weekday or month name are used, the caller
///      validates whether the value is valid. So the caller handles that
///      requirement of Table 97: Meaning of conversion specifiers
///      [tab:time.format.spec].
///
/// When no chrono-specs are provided it uses the stream formatter.

template <class _CharT, class _Tp>
_LIBCPP_HIDE_FROM_ABI void __format_chrono_using_chrono_specs(
    const _Tp& __value, basic_stringstream<_CharT>& __sstr, basic_string_view<_CharT> __chrono_specs) {
  tm __t              = std::__convert_to_tm(__value);
  const auto& __facet = std::use_facet<time_put<_CharT>>(__sstr.getloc());
  for (auto __it = __chrono_specs.begin(); __it != __chrono_specs.end(); ++__it) {
    if (*__it == _CharT('%')) {
      auto __s = __it;
      ++__it;
      // We only handle the types that can't be directly handled by time_put.
      // (as an optimization n, t, and % are also handled directly.)
      switch (*__it) {
      case _CharT('n'):
        __sstr << _CharT('\n');
        break;
      case _CharT('t'):
        __sstr << _CharT('\t');
        break;
      case _CharT('%'):
        __sstr << *__it;
        break;

      case _CharT('O'):
        ++__it;
        [[fallthrough]];
      default:
        __facet.put({__sstr}, __sstr, _CharT(' '), std::addressof(__t), __s, __it + 1);
        break;
      }
    } else {
      __sstr << *__it;
    }
  }
}

template <class _CharT, class _Tp>
_LIBCPP_HIDE_FROM_ABI auto
__format_chrono(const _Tp& __value,
                auto& __ctx,
                __format_spec::__parsed_specifications<_CharT> __specs,
                basic_string_view<_CharT> __chrono_specs) -> decltype(__ctx.out()) {
  basic_stringstream<_CharT> __sstr;
  // [time.format]/2
  // 2.1 - the "C" locale if the L option is not present in chrono-format-spec, otherwise
  // 2.2 - the locale passed to the formatting function if any, otherwise
  // 2.3 - the global locale.
  // Note that the __ctx's locale() call does 2.2 and 2.3.
  if (__specs.__chrono_.__locale_specific_form_)
    __sstr.imbue(__ctx.locale());
  else
    __sstr.imbue(locale::classic());

  if (__chrono_specs.empty())
    __sstr << __value;
  else
    __formatter::__format_chrono_using_chrono_specs(__value, __sstr, __chrono_specs);

  // TODO FMT Use the stringstream's view after P0408R7 has been implemented.
  basic_string<_CharT> __str = __sstr.str();
  return __formatter::__write_string(basic_string_view<_CharT>{__str}, __ctx.out(), __specs);
}

} // namespace __formatter

template <__fmt_char_type _CharT>
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT __formatter_chrono {
public:
  _LIBCPP_HIDE_FROM_ABI constexpr auto __parse(
      basic_format_parse_context<_CharT>& __parse_ctx, __format_spec::__fields __fields, __format_spec::__flags __flags)
      -> decltype(__parse_ctx.begin()) {
    return __parser_.__parse(__parse_ctx, __fields, __flags);
  }

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI auto format(const _Tp& __value, auto& __ctx) const -> decltype(__ctx.out()) const {
    return __formatter::__format_chrono(
        __value, __ctx, __parser_.__parser_.__get_parsed_chrono_specifications(__ctx), __parser_.__chrono_specs_);
  }

  __format_spec::__parser_chrono<_CharT> __parser_;
};

template <__fmt_char_type _CharT>
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT formatter<chrono::day, _CharT>
    : public __formatter_chrono<_CharT> {
public:
  using _Base = __formatter_chrono<_CharT>;

  _LIBCPP_HIDE_FROM_ABI constexpr auto parse(basic_format_parse_context<_CharT>& __parse_ctx)
      -> decltype(__parse_ctx.begin()) {
    return _Base::__parse(__parse_ctx, __format_spec::__fields_chrono, __format_spec::__flags::__day);
  }
};

#endif //if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_FORMAT)

_LIBCPP_END_NAMESPACE_STD

#endif //  _LIBCPP___CHRONO_FORMATTER_H
