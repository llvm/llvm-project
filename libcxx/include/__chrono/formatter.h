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
#include <__chrono/statically_widen.h>
#include <__chrono/year.h>
#include <__config>
#include <__format/concepts.h>
#include <__format/format_functions.h>
#include <__format/format_parse_context.h>
#include <__format/formatter.h>
#include <__format/formatter_output.h>
#include <__format/parser_std_format_spec.h>
#include <__memory/addressof.h>
#include <cmath>
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

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI void __format_year(int __year, basic_stringstream<_CharT>& __sstr) {
  if (__year < 0) {
    __sstr << _CharT('-');
    __year = -__year;
  }

  // TODO FMT Write an issue
  //   If the result has less than four digits it is zero-padded with 0 to two digits.
  // is less -> has less
  // left-padded -> zero-padded, otherwise the proper value would be 000-0.

  // Note according to the wording it should be left padded, which is odd.
  __sstr << std::format(_LIBCPP_STATICALLY_WIDEN(_CharT, "{:04}"), __year);
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI void __format_century(int __year, basic_stringstream<_CharT>& __sstr) {
  // TODO FMT Write an issue
  // [tab:time.format.spec]
  //   %C The year divided by 100 using floored division. If the result is a
  //   single decimal digit, it is prefixed with 0.

  bool __negative = __year < 0;
  int __century   = (__year - (99 * __negative)) / 100; // floored division
  __sstr << std::format(_LIBCPP_STATICALLY_WIDEN(_CharT, "{:02}"), __century);
}

template <class _CharT, class _Tp>
_LIBCPP_HIDE_FROM_ABI void __format_chrono_using_chrono_specs(
    const _Tp& __value, basic_stringstream<_CharT>& __sstr, basic_string_view<_CharT> __chrono_specs) {
  tm __t              = std::__convert_to_tm<tm>(__value);
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

      case _CharT('C'): {
        // strftime's output is only defined in the range [00, 99].
        int __year = __t.tm_year + 1900;
        if (__year < 1000 || __year > 9999)
          __formatter::__format_century(__year, __sstr);
        else
          __facet.put({__sstr}, __sstr, _CharT(' '), std::addressof(__t), __s, __it + 1);
      } break;

      // Unlike time_put and strftime the formatting library requires %Y
      //
      // [tab:time.format.spec]
      //   The year as a decimal number. If the result is less than four digits
      //   it is left-padded with 0 to four digits.
      //
      // This means years in the range (-1000, 1000) need manual formatting.
      // It's unclear whether %EY needs the same treatment. For example the
      // Japanese EY contains the era name and year. This is zero-padded to 2
      // digits in time_put (note that older glibc versions didn't do
      // padding.) However most eras won't reach 100 years, let alone 1000.
      // So padding to 4 digits seems unwanted for Japanese.
      //
      // The same applies to %Ex since that too depends on the era.
      //
      // %x the locale's date representation is currently doesn't handle the
      // zero-padding too.
      //
      // The 4 digits can be implemented better at a later time. On POSIX
      // systems the required information can be extracted by nl_langinfo
      // https://man7.org/linux/man-pages/man3/nl_langinfo.3.html
      //
      // Note since year < -1000 is expected to be rare it uses the more
      // expensive year routine.
      //
      // TODO FMT evaluate the comment above.

#  if defined(__GLIBC__) || defined(_AIX)
      case _CharT('y'):
        // Glibc fails for negative values, AIX for positive values too.
        __sstr << std::format(_LIBCPP_STATICALLY_WIDEN(_CharT, "{:02}"), (std::abs(__t.tm_year + 1900)) % 100);
        break;
#  endif // defined(__GLIBC__) || defined(_AIX)

      case _CharT('Y'): {
        int __year = __t.tm_year + 1900;
        if (__year < 1000)
          __formatter::__format_year(__year, __sstr);
        else
          __facet.put({__sstr}, __sstr, _CharT(' '), std::addressof(__t), __s, __it + 1);
      } break;

      case _CharT('O'):
      case _CharT('E'):
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

template <__fmt_char_type _CharT>
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT formatter<chrono::year, _CharT>
    : public __formatter_chrono<_CharT> {
public:
  using _Base = __formatter_chrono<_CharT>;

  _LIBCPP_HIDE_FROM_ABI constexpr auto parse(basic_format_parse_context<_CharT>& __parse_ctx)
      -> decltype(__parse_ctx.begin()) {
    return _Base::__parse(__parse_ctx, __format_spec::__fields_chrono, __format_spec::__flags::__year);
  }
};

#endif // if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_FORMAT)

_LIBCPP_END_NAMESPACE_STD

#endif //  _LIBCPP___CHRONO_FORMATTER_H
