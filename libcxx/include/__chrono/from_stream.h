// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_FROM_STREAM_H
#define _LIBCPP___CHRONO_FROM_STREAM_H

#include <__config>

#if _LIBCPP_HAS_LOCALIZATION

#  include <__chrono/calendar.h>
#  include <__chrono/concepts.h>
#  include <__chrono/day.h>
#  include <__chrono/duration.h>
#  include <__chrono/file_clock.h>
#  include <__chrono/gps_clock.h>
#  include <__chrono/hh_mm_ss.h>
#  include <__chrono/leap_second.h>
#  include <__chrono/month.h>
#  include <__chrono/monthday.h>
#  include <__chrono/parser_data.h>
#  include <__chrono/system_clock.h>
#  include <__chrono/tai_clock.h>
#  include <__chrono/time_point.h>
#  include <__chrono/tzdb.h>
#  include <__chrono/tzdb_list.h>
#  include <__chrono/utc_clock.h>
#  include <__chrono/weekday.h>
#  include <__chrono/year.h>
#  include <__chrono/year_month.h>
#  include <__chrono/year_month_day.h>
#  include <__fwd/memory.h>
#  include <__fwd/string.h>
#  include <__iterator/istreambuf_iterator.h>
#  include <__locale_dir/ctype.h>
#  include <__locale_dir/locale.h>
#  include <__locale_dir/num.h>
#  include <__locale_dir/time.h>
#  include <__memory/addressof.h>
#  include <__type_traits/common_type.h>
#  include <__utility/move.h>
#  include <cctype>
#  include <cstdint>
#  include <ctime>
#  include <istream>
#  include <limits>
#  include <ratio>
#  include <string>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_PUSH_MACROS
#  include <__undef_macros>

#  if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {
struct __parse_options {
  // The number of fractional digits used to determine the default width of %S.
  unsigned __default_fractional_width_ = 0;

  _LIBCPP_HIDE_FROM_ABI constexpr bool __allows_fraction() const { return __default_fractional_width_ != 0; }
};

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr __parse_options __get_parse_options() {
  if constexpr (__is_duration_v<_Tp>) {
    constexpr unsigned __width = hh_mm_ss<_Tp>::fractional_width;
    // Types such as duration<double> have fractional_width == 0 but can store 1.5 seconds.
    // Use __subsecond_precision as the default number of fractional digits for %S.
    if constexpr (__width == 0 && treat_as_floating_point_v<typename _Tp::rep>)
      return {__fields_storage::__subsecond_precision};
    else
      return {__width};
  } else if constexpr (__is_time_point<_Tp>) {
    return chrono::__get_parse_options<typename _Tp::duration>();
  } else {
    return {};
  }
}

// Check an inclusive range without narrowing parsed int or int64_t fields.
_LIBCPP_HIDE_FROM_ABI constexpr bool __in_range(int64_t __value, int64_t __lo, int64_t __hi) {
  return __lo <= __value && __value <= __hi;
}

_LIBCPP_HIDE_FROM_ABI constexpr int64_t __pow10(unsigned __exp) {
  int64_t __result = 1;
  for (unsigned __i = 0; __i < __exp; ++__i)
    __result *= 10;
  return __result;
}

struct __read_digits_result {
  uint64_t __value       = 0;
  unsigned __digits_read = 0;
  bool __overflow        = false;
};

_LIBCPP_HIDE_FROM_ABI constexpr bool __width_allowed(char __spec) {
  switch (__spec) {
  case 'C':
  case 'd':
  case 'e':
  case 'F':
  case 'g':
  case 'G':
  case 'H':
  case 'I':
  case 'j':
  case 'm':
  case 'M':
  case 'S':
  case 'u':
  case 'U':
  case 'V':
  case 'w':
  case 'W':
  case 'y':
  case 'Y':
    return true;
  default:
    return false;
  }
}

// Returns whether [time.parse] permits '__modifier' for '__spec'.
_LIBCPP_HIDE_FROM_ABI constexpr bool __modifier_allowed(char __modifier, char __spec) {
  if (__modifier == 'E')
    switch (__spec) {
    case 'c':
    case 'C':
    case 'x':
    case 'X':
    case 'y':
    case 'Y':
    case 'z':
      return true;
    default:
      return false;
    }

  switch (__spec) {
  case 'd':
  case 'e':
  case 'H':
  case 'I':
  case 'm':
  case 'M':
  case 'S':
  case 'U':
  case 'w':
  case 'W':
  case 'y':
  case 'z':
    return true;
  default:
    return false;
  }
}

template <class _CharT, class _Traits>
class __from_stream_parser {
  // Share the input stream and local error state across all parsing operations.
  basic_istream<_CharT, _Traits>& __is_;
  ios_base::iostate& __state_;

public:
  _LIBCPP_HIDE_FROM_ABI __from_stream_parser(basic_istream<_CharT, _Traits>& __is, ios_base::iostate& __state) noexcept
      : __is_(__is), __state_(__state) {}

  _LIBCPP_HIDE_FROM_ABI bool __fail() const noexcept {
    return (__state_ & (ios_base::failbit | ios_base::badbit)) != 0;
  }

  _LIBCPP_HIDE_FROM_ABI bool __peek(_CharT& __c) {
    if (__state_ != ios_base::goodbit)
      return false;

    typename _Traits::int_type __i = __is_.rdbuf()->sgetc();
    if (_Traits::eq_int_type(__i, _Traits::eof())) {
      __state_ |= ios_base::eofbit;
      return false;
    }

    __c = _Traits::to_char_type(__i);
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI void __consume() {
    // A character is required, but the input has already reached EOF.
    if (__state_ & ios_base::eofbit) {
      __state_ |= ios_base::failbit;
      return;
    }

    const auto __i = __is_.rdbuf()->sbumpc();
    if (_Traits::eq_int_type(__i, _Traits::eof()))
      __state_ |= ios_base::eofbit | ios_base::failbit;
  }

  // Extracts digits and reports their value, count, and overflow status.
  // Stops after consuming the first digit that would overflow '__max_value'.
  _LIBCPP_HIDE_FROM_ABI __read_digits_result __read_digits(unsigned __max_digits, uint64_t __max_value) {
    uint64_t __result      = 0;
    unsigned __digits_read = 0;

    for (_CharT __c{}; __digits_read < __max_digits && __peek(__c);) {
      if (__c < _CharT('0') || __c > _CharT('9'))
        break;
      __consume();
      ++__digits_read;

      const uint64_t __digit = static_cast<uint64_t>(__c - _CharT('0'));
      if (__result > __max_value / 10 || (__result == __max_value / 10 && __digit > __max_value % 10))
        return {__result, __digits_read, true};
      __result = __result * 10 + __digit;
    }

    return {__result, __digits_read, false};
  }

  // Reads an integer without a sign into int. Returns the number of digits read.
  // Failure leaves '__value' unchanged.
  _LIBCPP_HIDE_FROM_ABI unsigned __read_unsigned(unsigned __max_digits, int& __value) {
    auto __result = __read_digits(__max_digits, (numeric_limits<int>::max)());

    if (__result.__digits_read == 0 || __result.__overflow)
      __state_ |= ios_base::failbit;
    else
      __value = static_cast<int>(__result.__value);

    return __result.__digits_read;
  }

  // Reads an integer with an optional '+' or '-'. Failure leaves '__value' unchanged.
  // The width includes an optional sign.
  _LIBCPP_HIDE_FROM_ABI void __read_signed(unsigned __width, int& __value) {
    bool __negative = false;
    if (_CharT __c{}; __width != 0 && __peek(__c) && (_Traits::eq(__c, _CharT('-')) || _Traits::eq(__c, _CharT('+')))) {
      __negative = _Traits::eq(__c, _CharT('-'));
      __consume();
      --__width;
    }

    const uint64_t __positive_limit = static_cast<uint64_t>((numeric_limits<int>::max)());
    const uint64_t __negative_limit = __positive_limit + 1;
    const uint64_t __limit          = __negative ? __negative_limit : __positive_limit;

    auto __result = __read_digits(__width, __limit);
    if (__result.__digits_read == 0 || __result.__overflow) {
      __state_ |= ios_base::failbit;
      return;
    }

    if (__negative)
      __value = static_cast<int>(-static_cast<int64_t>(__result.__value));
    else
      __value = static_cast<int>(__result.__value);
  }

  // Parses one locale-dependent conversion specifier with time_get.
  _LIBCPP_HIDE_FROM_ABI bool __read_with_time_get(tm& __tm, char __spec, char __modifier = 0) {
    using _Iter  = istreambuf_iterator<_CharT, _Traits>;
    using _Facet = time_get<_CharT, _Iter>;

    const _Facet& __tf = std::use_facet<_Facet>(__is_.getloc());
    __tf.get(_Iter(__is_), _Iter(), __is_, __state_, std::addressof(__tm), __spec, __modifier);

    // time_get may report only eofbit for incomplete locale formats
    // (e.g. "01:30" parsed with %X in the classic locale), causing truncated
    // input to be accepted here. Fix format completeness checking in time_get.
    //
    // Propagate eofbit as well as parse failures. Reaching EOF after a
    // successful match is not itself a parsing failure.
    return !__fail();
  }

  // Parses a locale-dependent month name into a one-based month.
  _LIBCPP_HIDE_FROM_ABI void __read_month_name(int& __value) {
    tm __tm{};
    if (__read_with_time_get(__tm, 'b')) {
      if (__tm.tm_mon == (numeric_limits<int>::max)())
        __state_ |= ios_base::failbit;
      else
        __value = __tm.tm_mon + 1; // tm_mon is 0-based [0, 11].
    }
  }

  // Parses a locale-dependent weekday name using the chrono::weekday convention.
  _LIBCPP_HIDE_FROM_ABI void __read_weekday_name(int& __value) {
    tm __tm{};
    if (__read_with_time_get(__tm, 'a'))
      __value = __tm.tm_wday; // tm_wday is already [0, 6], Sunday == 0.
  }

  // time_get reports %p through tm_hour. Starting at zero leaves AM as 0
  // and changes PM to 12.
  _LIBCPP_HIDE_FROM_ABI void __read_am_pm(bool& __is_pm) {
    tm __tm{};
    if (__read_with_time_get(__tm, 'p'))
      __is_pm = __tm.tm_hour == 12;
  }

  // Parses an O-modified field through time_get and converts its tm member.
  _LIBCPP_HIDE_FROM_ABI void __read_alternative_field(char __spec, int& __field) {
    tm __tm{};
    if (!__read_with_time_get(__tm, __spec, 'O'))
      return;

    switch (__spec) {
    case 'd':
    case 'e':
      __field = __tm.tm_mday;
      break;
    case 'm':
      if (__tm.tm_mon == (numeric_limits<int>::max)())
        __state_ |= ios_base::failbit;
      else
        __field = __tm.tm_mon + 1;
      break;
    case 'H':
      __field = __tm.tm_hour;
      break;
    case 'I':
      // Keep the 12-hour value separate from %p until the fields are combined.
      __field = __tm.tm_hour == 0 ? 12 : __tm.tm_hour;
      break;
    case 'M':
      __field = __tm.tm_min;
      break;
    case 'w':
      __field = __tm.tm_wday;
      break;
    case 'y': {
      // tm_year is relative to 1900; %Oy supplies only the last two digits.
      int64_t __year = static_cast<int64_t>(__tm.tm_year) + 1900;
      __field        = static_cast<int>((__year < 0 ? -__year : __year) % 100);
      break;
    }
    default:
      __state_ |= ios_base::failbit;
      return;
    }
  }

  // Parses %S and its optional fractional part. '__width' includes the decimal point.
  _LIBCPP_HIDE_FROM_ABI void
  __read_seconds(unsigned __width, bool __allow_fraction, int& __seconds, int64_t& __subseconds) {
    constexpr unsigned __precision = __fields_storage::__subsecond_precision;
    const auto __integer           = __read_digits(__width, (numeric_limits<int>::max)());
    if (__integer.__overflow) {
      __state_ |= ios_base::failbit;
      return;
    }

    unsigned __remaining = __width - __integer.__digits_read;
    __read_digits_result __fraction{};
    _CharT __c{};
    const _CharT __decimal_point = std::use_facet<numpunct<_CharT> >(__is_.getloc()).decimal_point();
    if (__allow_fraction && __remaining != 0 && __peek(__c) && _Traits::eq(__c, __decimal_point)) {
      __consume();
      --__remaining;
      __fraction =
          __read_digits(__remaining < __precision ? __remaining : __precision, (numeric_limits<int64_t>::max)());
      __remaining -= __fraction.__digits_read;
      // Consume excess fractional digits within the field width, discarding them without rounding.
      while (__remaining != 0 && __peek(__c) && _CharT('0') <= __c && __c <= _CharT('9')) {
        __consume();
        --__remaining;
      }
    }

    // Either side of the decimal point may be empty, but at least one digit is required.
    if (__integer.__digits_read == 0 && __fraction.__digits_read == 0) {
      __state_ |= ios_base::failbit;
      return;
    }

    // The fraction is stored scaled to attoseconds, so that the conversion to the
    // target's precision does not depend on the number of digits that were read.
    __seconds    = static_cast<int>(__integer.__value);
    __subseconds = static_cast<int64_t>(__fraction.__value) * chrono::__pow10(__precision - __fraction.__digits_read);
  }

  // Parses %z as [+|-]hh[mm], and %Ez/%Oz as [+|-]h[h][:mm].
  _LIBCPP_HIDE_FROM_ABI void __read_utc_offset(bool __is_modified, int& __value) {
    _CharT __c{};
    if (!__peek(__c)) {
      __state_ |= ios_base::failbit;
      return;
    }
    int __sign = 1;
    if (_Traits::eq(__c, _CharT('-'))) {
      __sign = -1;
      __consume();
    } else if (_Traits::eq(__c, _CharT('+'))) {
      __consume();
    }

    int __hours            = 0;
    unsigned __digits_read = __read_unsigned(2, __hours);

    // %z requires exactly two hour digits, while %Ez and %Oz allow one or two.
    if (__fail() || (!__is_modified && __digits_read != 2)) {
      __state_ |= ios_base::failbit;
      return;
    }

    // The minutes are optional in both forms, but the modified form requires the
    // colon before them, and a colon requires the minutes to follow.
    int __minutes = 0;
    bool __has_minutes{};
    if (__is_modified) {
      __has_minutes = __peek(__c) && _Traits::eq(__c, _CharT(':'));
      if (__has_minutes)
        __consume();
    } else {
      __has_minutes = __peek(__c) && __c >= _CharT('0') && __c <= _CharT('9');
    }

    if (__has_minutes && __read_unsigned(2, __minutes) != 2) {
      __state_ |= ios_base::failbit;
      return;
    }

    __value = __sign * (__hours * 60 + __minutes);
  }

  // Parses a nonempty %Z token containing alphanumerics or '_', '/', '-', and '+'.
  _LIBCPP_HIDE_FROM_ABI void __read_time_zone_abbrev(basic_string<_CharT, _Traits>& __abbrev) {
    const auto& __ctype = std::use_facet<ctype<_CharT> >(__is_.getloc());
    __abbrev.clear();

    for (_CharT __c{}; __peek(__c);) {
      char __narrow = __ctype.narrow(__c, '\0');
      if (!std::isdigit(static_cast<unsigned char>(__narrow)) && ('a' > __narrow || __narrow > 'z') &&
          ('A' > __narrow || __narrow > 'Z') && __narrow != '_' && __narrow != '/' && __narrow != '-' &&
          __narrow != '+')
        break;

      __consume();
      __abbrev.push_back(__c);
    }

    if (__abbrev.empty())
      __state_ |= ios_base::failbit;
  }

  // Parses '__fmt' into '__f', setting failbit on a mismatch.
  // Keep the handling of all format specifiers together in a single switch.
  // This makes the function longer and increases its cognitive complexity.
  // NOLINTNEXTLINE(readability-function-size, readability-function-cognitive-complexity)
  _LIBCPP_HIDE_FROM_ABI void __parse(
      const _CharT* __fmt, __fields_storage& __f, basic_string<_CharT, _Traits>& __abbrev, __parse_options __options) {
    const auto& __ctype = std::use_facet<ctype<_CharT> >(__is_.getloc());

    auto __skip_one_whitespace = [&] {
      _CharT __c{};
      if (!__peek(__c) || !__ctype.is(ctype_base::space, __c))
        return false;
      __consume();
      return true;
    };

    auto __match = [&](_CharT __expected) {
      _CharT __c{};
      if (!__peek(__c) || !_Traits::eq(__c, __expected)) {
        __state_ |= ios_base::failbit;
        return false;
      }
      __consume();
      return !__fail();
    };

    // Accept repeated fields only when their values agree.
    auto __assign = [&](auto& __field, auto __value, __fields_set __field_bit) {
      if (__f.__has(__field_bit) && __field != __value) {
        __state_ |= ios_base::failbit;
        return false;
      }
      __field = __value;
      __f.__set(__field_bit);
      return true;
    };

    auto __assign_seconds = [&](int __seconds, int64_t __subseconds) {
      // Repeated seconds fields must agree in both their integer and fractional parts.
      if (__f.__has(__fields_set::__seconds) && (__f.__seconds_ != __seconds || __f.__subseconds_ != __subseconds)) {
        __state_ |= ios_base::failbit;
        return false;
      }

      __f.__seconds_    = __seconds;
      __f.__subseconds_ = __subseconds;
      __f.__set(__fields_set::__seconds);
      return true;
    };

    auto __assign_date = [&](const tm& __tm) {
      const int64_t __year = static_cast<int64_t>(__tm.tm_year) + 1900;
      if (__year > (numeric_limits<int>::max)() || __tm.tm_mon == (numeric_limits<int>::max)()) {
        __state_ |= ios_base::failbit;
        return false;
      }
      return __assign(__f.__year_, static_cast<int>(__year), __fields_set::__year) &&
             __assign(__f.__month_, __tm.tm_mon + 1, __fields_set::__month) &&
             __assign(__f.__day_, __tm.tm_mday, __fields_set::__day) &&
             (__tm.tm_wday == -1 || __assign(__f.__weekday_, __tm.tm_wday, __fields_set::__weekday));
    };

    auto __assign_time = [&](const tm& __tm) {
      return __assign(__f.__hours_, __tm.tm_hour, __fields_set::__hours) &&
             __assign(__f.__minutes_, __tm.tm_min, __fields_set::__minutes) && __assign_seconds(__tm.tm_sec, 0);
    };

    while (*__fmt != _CharT('\0')) {
      if (__fail())
        return;

      if (__ctype.is(ctype_base::space, *__fmt)) {
        while (__skip_one_whitespace()) {
        }
        ++__fmt;
        continue;
      }

      if (*__fmt != _CharT('%')) {
        __match(*__fmt);
        ++__fmt;
        continue;
      }

      ++__fmt;

      unsigned __width = 0;
      bool __has_width = false;
      while (_CharT('0') <= *__fmt && *__fmt <= _CharT('9')) {
        __has_width            = true;
        const unsigned __digit = static_cast<unsigned>(*__fmt - _CharT('0'));
        if (__width > ((numeric_limits<unsigned>::max)() - __digit) / 10) {
          __state_ |= ios_base::failbit;
          return;
        }
        __width = __width * 10 + __digit;
        ++__fmt;
      }

      char __modifier = '\0';
      if (*__fmt == _CharT('E') || *__fmt == _CharT('O')) {
        __modifier = static_cast<char>(*__fmt == _CharT('E') ? 'E' : 'O');
        ++__fmt;
      }

      if (__has_width && __modifier != '\0') {
        __state_ |= ios_base::failbit;
        return;
      }

      char __spec = __ctype.narrow(*__fmt, '\0');

      if ((__has_width && !__width_allowed(__spec)) || (__modifier != 0 && !__modifier_allowed(__modifier, __spec))) {
        __state_ |= ios_base::failbit;
        return;
      }

      int __value{};
      switch (__spec) {
      case 'a':
      case 'A':
        __read_weekday_name(__value);
        if (!__fail())
          __assign(__f.__weekday_, __value, __fields_set::__weekday);
        break;

      case 'b':
      case 'h':
      case 'B':
        __read_month_name(__value);
        if (!__fail())
          __assign(__f.__month_, __value, __fields_set::__month);
        break;

      case 'c': {
        tm __tm{};
        // The locale's format may omit the weekday. Use -1 to distinguish an
        // unparsed weekday from Sunday, whose tm_wday value is 0.
        __tm.tm_wday = -1;
        if (__read_with_time_get(__tm, __spec, __modifier) && __assign_date(__tm))
          __assign_time(__tm);
        break;
      }

      case 'C':
        if (__modifier == 'E') {
          // time_get does not support %C; parse %EC numerically.
          __read_signed(2, __value);
        } else {
          __read_signed(__has_width ? __width : 2, __value);
        }
        if (!__fail())
          __assign(__f.__century_, __value, __fields_set::__century);
        break;

      case 'd':
      case 'e':
        if (__modifier == 'O')
          __read_alternative_field(__spec, __value);
        else
          __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail())
          __assign(__f.__day_, __value, __fields_set::__day);
        break;

      case 'D':
        __read_unsigned(2, __value);
        if (__fail() || !__assign(__f.__month_, __value, __fields_set::__month) || !__match(_CharT('/')))
          break;
        __read_unsigned(2, __value);
        if (__fail() || !__assign(__f.__day_, __value, __fields_set::__day) || !__match(_CharT('/')))
          break;
        __read_unsigned(2, __value);
        if (!__fail())
          __assign(__f.__year_of_century_, __value, __fields_set::__year_of_century);
        break;

      case 'F':
        // A width on %F applies only to %Y.
        __read_signed(__has_width ? __width : 4, __value);
        if (__fail() || !__assign(__f.__year_, __value, __fields_set::__year) || !__match(_CharT('-')))
          break;
        __read_unsigned(2, __value);
        if (__fail() || !__assign(__f.__month_, __value, __fields_set::__month) || !__match(_CharT('-')))
          break;
        __read_unsigned(2, __value);
        if (!__fail())
          __assign(__f.__day_, __value, __fields_set::__day);
        break;

      case 'g':
        __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail()) {
          if (!__in_range(__value, 0, 99))
            __state_ |= ios_base::failbit;
          else
            __assign(__f.__iso_year_of_century_, __value, __fields_set::__iso_year_of_century);
        }
        break;

      case 'G':
        __read_signed(__has_width ? __width : 4, __value);
        if (!__fail())
          __assign(__f.__iso_year_, __value, __fields_set::__iso_year);
        break;

      case 'H':
        if (__modifier == 'O')
          __read_alternative_field(__spec, __value);
        else
          __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail())
          __assign(__f.__hours_, __value, __fields_set::__hours);
        break;

      case 'I':
        if (__modifier == 'O')
          __read_alternative_field(__spec, __value);
        else
          __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail())
          __assign(__f.__hour12_, __value, __fields_set::__hour12);
        break;

      case 'j':
        // The day of the year for a calendar type; a plain number of days when
        // the target is a duration, in which case it is not limited to [1, 366].
        __read_unsigned(__has_width ? __width : 3, __value);
        if (!__fail())
          __assign(__f.__day_of_year_, __value, __fields_set::__day_of_year);
        break;

      case 'm':
        if (__modifier == 'O')
          __read_alternative_field(__spec, __value);
        else
          __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail())
          __assign(__f.__month_, __value, __fields_set::__month);
        break;

      case 'M':
        if (__modifier == 'O')
          __read_alternative_field(__spec, __value);
        else
          __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail())
          __assign(__f.__minutes_, __value, __fields_set::__minutes);
        break;

      case 'p': {
        bool __is_pm{};
        __read_am_pm(__is_pm);
        if (!__fail())
          __assign(__f.__is_pm_, __is_pm, __fields_set::__am_pm);
        break;
      }

      case 'r':
      case 'X': {
        tm __tm{};
        if (__read_with_time_get(__tm, __spec, __modifier))
          __assign_time(__tm);
        break;
      }

      case 'R': // %H:%M
      case 'T': // %H:%M:%S
        __read_unsigned(2, __value);
        if (__fail() || !__assign(__f.__hours_, __value, __fields_set::__hours) || !__match(_CharT(':')))
          break;
        __read_unsigned(2, __value);
        if (__fail() || !__assign(__f.__minutes_, __value, __fields_set::__minutes))
          break;
        if (__spec == 'R')
          break;
        if (!__match(_CharT(':')))
          break;
        // %T has no width or modifier; parse its seconds as an ordinary %S.
        [[fallthrough]];

      case 'S': {
        // The target type determines whether fractions are allowed; the width only
        // limits the number of characters consumed.
        unsigned __default_width = __options.__allows_fraction() ? 3 + __options.__default_fractional_width_ : 2;
        // time_get ignores the O modifier; parse %OS using ordinary digits.
        int64_t __subseconds{};
        __read_seconds(__has_width ? __width : __default_width, __options.__allows_fraction(), __value, __subseconds);
        if (!__fail())
          __assign_seconds(__value, __subseconds);
        break;
      }

      case 'u': {
        int __weekday = 0;
        __read_unsigned(__has_width ? __width : 1, __weekday);
        if (!__fail()) {
          if (!__in_range(__weekday, 1, 7))
            __state_ |= ios_base::failbit;
          else
            __assign(__f.__weekday_, __weekday % 7, __fields_set::__weekday);
        }
        break;
      }

      case 'w': {
        if (__modifier == 'O')
          __read_alternative_field(__spec, __value);
        else
          __read_unsigned(__has_width ? __width : 1, __value);
        if (!__fail())
          __assign(__f.__weekday_, __value, __fields_set::__weekday);
        break;
      }

      case 'U':
        // time_get does not support %U; parse %OU numerically.
        __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail())
          __assign(__f.__week_sun_, __value, __fields_set::__week_sun);
        break;

      case 'V':
        __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail())
          __assign(__f.__iso_week_, __value, __fields_set::__iso_week);
        break;

      case 'W':
        // time_get does not support %W; parse %OW numerically.
        __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail())
          __assign(__f.__week_mon_, __value, __fields_set::__week_mon);
        break;

      case 'x': {
        tm __tm{};
        // The locale's format may omit the weekday. Use -1 to distinguish an
        // unparsed weekday from Sunday, whose tm_wday value is 0.
        __tm.tm_wday = -1;
        if (__read_with_time_get(__tm, __spec, __modifier))
          __assign_date(__tm);
        break;
      }

      case 'y':
        if (__modifier == 'O')
          __read_alternative_field(__spec, __value);
        else if (__modifier == 'E') {
          tm __tm{};
          if (__read_with_time_get(__tm, __spec, __modifier)) {
            const int64_t __year = static_cast<int64_t>(__tm.tm_year) + 1900;
            __value              = static_cast<int>((__year < 0 ? -__year : __year) % 100);
          }
        } else
          __read_unsigned(__has_width ? __width : 2, __value);
        if (!__fail())
          __assign(__f.__year_of_century_, __value, __fields_set::__year_of_century);
        break;

      case 'Y':
        if (__modifier == 'E') {
          tm __tm{};
          if (__read_with_time_get(__tm, __spec, __modifier)) {
            const int64_t __year = static_cast<int64_t>(__tm.tm_year) + 1900;
            if (__year > (numeric_limits<int>::max)())
              __state_ |= ios_base::failbit;
            else
              __value = static_cast<int>(__year);
          }
        } else
          __read_signed(__has_width ? __width : 4, __value);
        if (!__fail())
          __assign(__f.__year_, __value, __fields_set::__year);
        break;

      case 'z':
        __read_utc_offset(__modifier != 0, __value);
        if (!__fail())
          __assign(__f.__utc_offset_, __value, __fields_set::__utc_offset);
        break;

      case 'Z': {
        basic_string<_CharT, _Traits> __parsed;
        __read_time_zone_abbrev(__parsed);
        if (!__fail()) {
          // A parsed abbreviation is nonempty; an empty string means no prior %Z.
          if (!__abbrev.empty() && __abbrev != __parsed)
            __state_ |= ios_base::failbit;
          else
            __abbrev = std::move(__parsed);
        }
        break;
      }

      case 'n':
        // %n matches exactly one white space character, %t at most one. Combining
        // them and a literal space matches a range, e.g. "%n%t%t" matches one to
        // three white space characters.
        if (!__skip_one_whitespace())
          __state_ |= ios_base::failbit;
        break;
      case 't':
        __skip_one_whitespace();
        break;
      case '%':
        __match(_CharT('%'));
        break;
      default:
        __state_ |= ios_base::failbit;
        return;
      }
      ++__fmt;
    }
  }
};

_LIBCPP_HIDE_FROM_ABI inline int64_t __compose_year(int __century, int __last_two_digits) {
  // %C uses floored division; %y and %g contain the absolute last two digits.
  return static_cast<int64_t>(__century) * 100 +
         (__century < 0 && __last_two_digits != 0 ? 100 - __last_two_digits : __last_two_digits);
}

// Construct a candidate year from %C/%y or %Y, then check consistency with any
// other supplied year fields. Store the result only if it is representable.
_LIBCPP_HIDE_FROM_ABI inline bool __try_get_year(const __fields_storage& __f, int& __result) {
  int64_t __year{};

  if (__f.__has(__fields_set::__year_of_century)) {
    if (!__in_range(__f.__year_of_century_, 0, 99))
      return false;

    // Use an explicit century or full year before falling back to the default century.
    if (__f.__has(__fields_set::__century)) {
      __year = __compose_year(__f.__century_, __f.__year_of_century_);
    } else if (__f.__has(__fields_set::__year)) {
      __year = __f.__year_;
      if ((__year < 0 ? -__year : __year) % 100 != __f.__year_of_century_)
        return false;
    } else {
      __year = (__f.__year_of_century_ <= 68 ? 2000 : 1900) + __f.__year_of_century_;
    }

    // Check the constructed year against %Y, if supplied.
    if (__f.__has(__fields_set::__year) && __f.__year_ != __year)
      return false;
  } else if (__f.__has(__fields_set::__year)) {
    __year = __f.__year_;

    // Check the year obtained from %Y against %C, if supplied.
    if (__f.__has(__fields_set::__century) && __f.__century_ != __year / 100 - (__year % 100 < 0))
      return false;
  } else {
    return false;
  }

  if (!__in_range(__year, static_cast<int>((year::min)()), static_cast<int>((year::max)())))
    return false;

  __result = static_cast<int>(__year);
  return true;
}

// Converts an ISO year (%G, or expanded %g), week (%V), and weekday (%u/%w)
// to sys_days.
_LIBCPP_HIDE_FROM_ABI inline bool __iso_week_to_sys_days(int64_t __g, int __v, weekday __wd, sys_days& __result) {
  const int __min_year = static_cast<int>((year::min)());
  const int __max_year = static_cast<int>((year::max)());
  if (!__in_range(__g, __min_year - 1, __max_year + 1) || !__in_range(__v, 1, 53))
    return false;

  const days __year_length{__g % 4 == 0 && (__g % 100 != 0 || __g % 400 == 0) ? 366 : 365};
  // An ISO year can extend one year beyond chrono::year's range. Anchor its
  // January 1 in a representable year before doing arithmetic in sys_days.
  sys_days __jan1;
  if (__g < __min_year)
    __jan1 = sys_days{(year::min)() / January / 1} - __year_length;
  else if (__g > __max_year)
    __jan1 = sys_days{(year::max)() / January / 1} + days{(year::max)().is_leap() ? 366 : 365};
  else
    __jan1 = sys_days{year{static_cast<int>(__g)} / January / 1};

  // ISO week 1 contains __g-01-04 and starts on Monday.
  // Compute days from 1970-01-01 to __g/__v/__wd in four parts:
  // 1. Days from 1970-01-01 to __g-01-04.
  // 2. Subtract the initial partial week from that week's Monday to __g-01-04.
  // 3. (__v - 1) complete weeks preceding the requested week.
  // 4. The final partial week from Monday to __wd.
  sys_days __jan4 = __jan1 + days{3};
  weekday __jan4_wd{__jan4};
  sys_days __week1_start = __jan4 - days{static_cast<int>(__jan4_wd.iso_encoding()) - 1};
  sys_days __date        = __week1_start + weeks{__v - 1} + days{static_cast<int>(__wd.iso_encoding()) - 1};

  // Reject a nonexistent week: the Thursday of the result's week must fall in
  // the ISO year '__g'.
  sys_days __thursday = __date + days{4 - static_cast<int>(__wd.iso_encoding())};
  if (__thursday < __jan1 || __thursday >= __jan1 + __year_length)
    return false;

  // Check before converting to year_month_day, which could wrap the year.
  if (__date < sys_days{(year::min)() / January / 1} || __date > sys_days{(year::max)() / December / 31})
    return false;

  __result = __date;
  return true;
}

// Converts a calendar year (%Y or %C/%y), week (%U/%W), and weekday (%u/%w)
// to sys_days, including week zero.
// The caller must supply a valid year.
_LIBCPP_HIDE_FROM_ABI inline bool
__week_to_sys_days(int __year, int __week, weekday __first, weekday __wd, sys_days& __result) {
  if (!__in_range(__week, 0, 53))
    return false;

  // Compute days from 1970-01-01 to __year/__week/__wd in four parts:
  // 1. Days from 1970-01-01 to __year-01-01.
  // 2. The initial partial week from __year-01-01 to the next week start (__first).
  // 3. (__week - 1) complete weeks preceding the requested week.
  // 4. The final partial week from the week's start (__first) to __wd.
  sys_days __jan1{year{__year} / January / 1};
  sys_days __week1_start = __jan1 + (__first - weekday{__jan1});
  sys_days __date        = __week1_start + weeks{__week - 1} + (__wd - __first);
  if (year_month_day{__date}.year() != year{__year})
    return false;

  __result = __date;
  return true;
}

// Construct a valid date from the first complete combination of parsed fields.
_LIBCPP_HIDE_FROM_ABI inline bool __make_date(const __fields_storage& __f, year_month_day& __result) {
  // Calendar-year combinations use %Y or a year obtained from %C/%y.
  if (int __year{}; __try_get_year(__f, __year)) {
    // Calendar date: %Y %m %d, also supplied by formats such as %F, %D, or %x.
    if (__f.__has(__fields_set::__month | __fields_set::__day)) {
      if (!__in_range(__f.__month_, 1, 12) || !__in_range(__f.__day_, 1, 31))
        return false;

      const year_month_day __ymd{
          year{__year}, month{static_cast<unsigned>(__f.__month_)}, day{static_cast<unsigned>(__f.__day_)}};
      if (!__ymd.ok())
        return false;
      __result = __ymd;
      return true;
    }

    // Ordinal date: %Y %j.
    if (__f.__has(__fields_set::__day_of_year)) {
      if (!__in_range(__f.__day_of_year_, 1, year{__year}.is_leap() ? 366 : 365))
        return false;

      __result = year_month_day{sys_days{year{__year} / January / 1} + days{__f.__day_of_year_ - 1}};
      return true;
    }

    // Week date: %Y with %U or %W and a weekday (%a/%A/%u/%w).
    if (__f.__has(__fields_set::__weekday) && __f.__has_any(__fields_set::__week_sun | __fields_set::__week_mon)) {
      if (!__in_range(__f.__weekday_, 0, 6))
        return false;
      const bool __use_sunday = __f.__has(__fields_set::__week_sun);
      sys_days __date{};
      if (!__week_to_sys_days(
              __year,
              __use_sunday ? __f.__week_sun_ : __f.__week_mon_,
              __use_sunday ? Sunday : Monday,
              weekday{static_cast<unsigned>(__f.__weekday_)},
              __date))
        return false;

      __result = year_month_day{__date};
      return true;
    }
  }

  // ISO week date: %G (or expanded %g), %V, and a weekday (%a/%A/%u/%w).
  if (__f.__has(__fields_set::__iso_week | __fields_set::__weekday) &&
      __f.__has_any(__fields_set::__iso_year | __fields_set::__iso_year_of_century)) {
    // Use %G when supplied; expand %g only if needed to construct the date.
    int64_t __iso_year{};
    if (!__f.__has(__fields_set::__iso_year)) {
      int __century = __f.__iso_year_of_century_ <= 68 ? 20 : 19;
      if (__f.__has(__fields_set::__century))
        __century = __f.__century_;
      else if (__f.__has(__fields_set::__year))
        __century = __f.__year_ / 100 - (__f.__year_ % 100 < 0);
      else if (__f.__has(__fields_set::__year_of_century))
        __century = __f.__year_of_century_ <= 68 ? 20 : 19;
      __iso_year = __compose_year(__century, __f.__iso_year_of_century_);
    } else {
      __iso_year = __f.__iso_year_;
    }
    sys_days __date{};
    if (!__in_range(__f.__weekday_, 0, 6) ||
        !__iso_week_to_sys_days(__iso_year, __f.__iso_week_, weekday{static_cast<unsigned>(__f.__weekday_)}, __date))
      return false;

    __result = year_month_day{__date};
    return true;
  }

  return false;
}

// Compare all parsed date fields with a valid date, including fields that do
// not form a complete representation on their own, e.g. %F followed by only %G.
_LIBCPP_HIDE_FROM_ABI inline bool __validate_date(const __fields_storage& __f, const year_month_day& __ymd) {
  auto __matches = [&](__fields_set __field_bit, int __parsed, int __expected) {
    return !__f.__has(__field_bit) || __parsed == __expected;
  };

  // Calendar year: %Y, %C, and %y, including a standalone %C.
  const auto __year = static_cast<int>(__ymd.year());
  if (!__matches(__fields_set::__year, __f.__year_, __year) ||
      !__matches(__fields_set::__century, __f.__century_, __year / 100 - (__year % 100 < 0)) ||
      !__matches(__fields_set::__year_of_century, __f.__year_of_century_, (__year < 0 ? -__year : __year) % 100))
    return false;

  // Without %C or %Y, %y denotes a year in [1969, 2068], not just matching last digits.
  if (__f.__has(__fields_set::__year_of_century) && !__f.__has_any(__fields_set::__century | __fields_set::__year) &&
      !__in_range(__year, 1969, 2068))
    return false;

  // Month (%m/%b/%B/%h), day (%d/%e), and weekday (%a/%A/%u/%w).
  const sys_days __date{__ymd};
  const weekday __weekday{__date};
  if (!__matches(__fields_set::__month, __f.__month_, static_cast<unsigned>(__ymd.month())) ||
      !__matches(__fields_set::__day, __f.__day_, static_cast<unsigned>(__ymd.day())) ||
      !__matches(__fields_set::__weekday, __f.__weekday_, __weekday.c_encoding()))
    return false;

  // Day of year (%j) and Sunday-/Monday-based week numbers (%U/%W).
  const sys_days __jan1{__ymd.year() / January / 1};
  const int __day_of_year = static_cast<int>((__date - __jan1).count()) + 1;
  if (!__matches(__fields_set::__day_of_year, __f.__day_of_year_, __day_of_year) ||
      !__matches(__fields_set::__week_sun,
                 __f.__week_sun_,
                 (__day_of_year - static_cast<int>(__weekday.c_encoding()) + 6) / 7) ||
      !__matches(__fields_set::__week_mon,
                 __f.__week_mon_,
                 (__day_of_year - (static_cast<int>(__weekday.iso_encoding()) - 1) + 6) / 7))
    return false;

  if (__f.__has_any(__fields_set::__iso_year | __fields_set::__iso_year_of_century | __fields_set::__iso_week)) {
    // Find the Thursday of the week containing __date.
    // Its calendar year is the ISO year: adjust __iso_year and __iso_jan1
    // if that Thursday falls in the previous or next calendar year.
    // Compute the ISO week number from that Thursday's distance from __iso_jan1,
    // then compare the ISO year and week with any supplied %G/%g and %V fields.
    // Keep __iso_year as int because it can exceed chrono::year's valid range
    // near year::min()/max().
    const sys_days __thursday  = __date + days{4 - static_cast<int>(__weekday.iso_encoding())};
    const sys_days __next_jan1 = __jan1 + days{__ymd.year().is_leap() ? 366 : 365};
    int __iso_year             = __year;
    sys_days __iso_jan1        = __jan1;
    if (__thursday < __jan1) {
      // This week's Thursday falls in the previous calendar year, so the ISO year is __year - 1.
      --__iso_year;
      const bool __is_leap = __iso_year % 4 == 0 && (__iso_year % 100 != 0 || __iso_year % 400 == 0);
      __iso_jan1 -= days{__is_leap ? 366 : 365};
    } else if (__thursday >= __next_jan1) {
      ++__iso_year;
      __iso_jan1 = __next_jan1;
    }
    const int __iso_week = static_cast<int>((__thursday - __iso_jan1).count()) / 7 + 1;
    if (!__matches(__fields_set::__iso_year, __f.__iso_year_, __iso_year) ||
        !__matches(__fields_set::__iso_year_of_century,
                   __f.__iso_year_of_century_,
                   (__iso_year < 0 ? -__iso_year : __iso_year) % 100) ||
        !__matches(__fields_set::__iso_week, __f.__iso_week_, __iso_week))
      return false;
  }

  return true;
}

_LIBCPP_HIDE_FROM_ABI inline bool __try_get_date(const __fields_storage& __f, sys_days& __result) {
  // First construct a valid candidate, e.g. from %F/%x, %Y %j, or %G %V %u.
  year_month_day __ymd{};
  if (!__make_date(__f, __ymd))
    return false;

  // Then check all parsed date fields, including incomplete representations
  // such as a standalone %G or %V alongside %F.
  if (!__validate_date(__f, __ymd))
    return false;

  __result = sys_days{__ymd};
  return true;
}

// Validate and combine %H, %I, and %p, storing the hour only on success.
_LIBCPP_HIDE_FROM_ABI inline bool __try_get_hour(const __fields_storage& __f, int& __result) {
  int __hour{};
  if (__f.__has(__fields_set::__hours)) {
    // %H determines the hour; check agreement with %I and %p if supplied.
    __hour = __f.__hours_;
    if (!__in_range(__hour, 0, 23))
      return false;
    if (__f.__has(__fields_set::__hour12)) {
      const int __hour12 = static_cast<int>(chrono::make12(hours{__hour}).count());
      if (__f.__hour12_ != __hour12)
        return false;
    }
    if (__f.__has(__fields_set::__am_pm) && (__hour >= 12) != __f.__is_pm_)
      return false;
  } else if (__f.__has(__fields_set::__hour12)) {
    // Without %H, %I requires %p to distinguish AM from PM.
    if (!__in_range(__f.__hour12_, 1, 12) || !__f.__has(__fields_set::__am_pm))
      return false;
    __hour = static_cast<int>(chrono::make24(hours{__f.__hour12_}, __f.__is_pm_).count());
  } else {
    // Default to zero when neither hour field was supplied.
    __hour = 0;
  }

  __result = __hour;
  return true;
}

_LIBCPP_HIDE_FROM_ABI inline bool __validate_minute(const __fields_storage& __f) {
  return !__f.__has(__fields_set::__minutes) || __in_range(__f.__minutes_, 0, 59);
}

_LIBCPP_HIDE_FROM_ABI inline bool __validate_second(const __fields_storage& __f, int __max_second) {
  return (!__f.__has(__fields_set::__seconds) || __in_range(__f.__seconds_, 0, __max_second)) && __f.__subseconds_ >= 0;
}

// Validate the time-of-day fields and combine their whole seconds only on success.
_LIBCPP_HIDE_FROM_ABI inline bool
__try_get_time_of_day(const __fields_storage& __f, seconds& __result, int __max_second = 59) {
  int __hour{};
  if (!__try_get_hour(__f, __hour) || !__validate_minute(__f) || !__validate_second(__f, __max_second))
    return false;

  __result = hours{__hour} + minutes{__f.__minutes_} + seconds{__f.__seconds_};
  return true;
}

template <class _Duration, class _Unit>
_LIBCPP_HIDE_FROM_ABI constexpr bool __can_represent() {
  return treat_as_floating_point_v<typename _Duration::rep> ||
         ratio_less_equal_v<typename _Duration::period, typename _Unit::period>;
}

// Reject fields whose units are finer than the target precision.
template <class _Duration>
_LIBCPP_HIDE_FROM_ABI bool __validate_time_precision(const __fields_storage& __f) {
  if constexpr (!chrono::__can_represent<_Duration, days>())
    return false;

  if constexpr (!chrono::__can_represent<_Duration, hours>()) {
    if (__f.__has_any(__fields_set::__hours | __fields_set::__hour12 | __fields_set::__am_pm))
      return false;
  }

  if constexpr (!chrono::__can_represent<_Duration, minutes>()) {
    if (__f.__has(__fields_set::__minutes))
      return false;
  }

  if constexpr (!chrono::__can_represent<_Duration, seconds>()) {
    if (__f.__has(__fields_set::__seconds))
      return false;
  }

  return true;
}

// Convert whole seconds and a nonnegative attosecond fraction together.
// Overflow in duration conversion and accumulation is not checked.
template <class _Duration>
_LIBCPP_HIDE_FROM_ABI _Duration __make_duration(int64_t __seconds, int64_t __subseconds) {
  if constexpr (treat_as_floating_point_v<typename _Duration::rep>) {
    constexpr auto __scale = chrono::__pow10(__fields_storage::__subsecond_precision);
    const duration<long double> __value{
        static_cast<long double>(__seconds) + static_cast<long double>(__subseconds) / __scale};
    return chrono::duration_cast<_Duration>(__value);
  } else {
    using _HMS         = hh_mm_ss<_Duration>;
    using _Precision   = typename _HMS::precision;
    _Precision __value = chrono::duration_cast<_Precision>(seconds{__seconds});
    __value += _Precision{static_cast<typename _Precision::rep>(
        __subseconds / chrono::__pow10(__fields_storage::__subsecond_precision - _HMS::fractional_width))};
    return chrono::duration_cast<_Duration>(__value);
  }
}

// Builders validate parsed fields and convert them to the requested type.

template <class _Rep, class _Period>
_LIBCPP_HIDE_FROM_ABI bool __from_fields(const __fields_storage& __f, duration<_Rep, _Period>& __result) {
  if (!chrono::__validate_time_precision<duration<_Rep, _Period>>(__f))
    return false;

  // Durations can represent only elapsed days and time-of-day fields.
  // UTC offsets and time zone abbreviations are allowed but do not contribute to the duration.
  constexpr auto __duration_components =
      __fields_set::__day_of_year | __fields_set::__hours | __fields_set::__hour12 | __fields_set::__minutes |
      __fields_set::__seconds;
  if (!__f.__has_only(__duration_components | __fields_set::__am_pm))
    return false;

  // Require at least one numeric component; %p alone is insufficient.
  if (!__f.__has_any(__duration_components))
    return false;

  seconds __time_of_day{};
  // Use clock-time ranges for %H, %M, and %S; %j supplies any additional days.
  if (__f.__day_of_year_ < 0 || !__try_get_time_of_day(__f, __time_of_day))
    return false;

  // Sum the components in seconds; when parsing a duration, %j denotes a day count, not a day of the year.
  const seconds __whole_seconds = chrono::duration_cast<seconds>(days{__f.__day_of_year_}) + __time_of_day;

  __result = chrono::__make_duration<duration<_Rep, _Period>>(__whole_seconds.count(), __f.__subseconds_);
  return true;
}

template <class _Duration>
_LIBCPP_HIDE_FROM_ABI bool __from_fields(const __fields_storage& __f, sys_time<_Duration>& __result) {
  if (!chrono::__validate_time_precision<_Duration>(__f))
    return false;

  sys_days __date{};
  if (!__try_get_date(__f, __date))
    return false;

  // sys_time does not represent leap seconds, so seconds must be in [0, 59].
  seconds __time_of_day{};
  if (!__try_get_time_of_day(__f, __time_of_day))
    return false;

  // %z gives the offset of the parsed time from UTC, so it is subtracted to
  // arrive at the UTC time sys_time holds. It is zero when %z was not used.
  // Combine whole seconds before converting to avoid overflowing conversion
  // ratios from days or minutes to fine target precisions.
  const seconds __whole_seconds = __date.time_since_epoch() + __time_of_day - minutes{__f.__utc_offset_};
  using _Precision              = common_type_t<_Duration, seconds>;
  __result                      = sys_time<_Duration>{
      chrono::floor<_Duration>(chrono::__make_duration<_Precision>(__whole_seconds.count(), __f.__subseconds_))};
  return true;
}

// A parsed UTC offset is not applied to local_time.
template <class _Duration>
_LIBCPP_HIDE_FROM_ABI bool __from_fields(const __fields_storage& __f, local_time<_Duration>& __result) {
  if (!chrono::__validate_time_precision<_Duration>(__f))
    return false;

  sys_days __date{};
  if (!__try_get_date(__f, __date))
    return false;

  seconds __time_of_day{};
  if (!__try_get_time_of_day(__f, __time_of_day))
    return false;

  // Combine whole seconds before converting to the target precision.
  const seconds __whole_seconds = __date.time_since_epoch() + __time_of_day;
  using _Precision              = common_type_t<_Duration, seconds>;
  __result                      = local_time<_Duration>{
      chrono::floor<_Duration>(chrono::__make_duration<_Precision>(__whole_seconds.count(), __f.__subseconds_))};
  return true;
}

template <class _Duration>
_LIBCPP_HIDE_FROM_ABI bool __from_fields(const __fields_storage& __f, file_time<_Duration>& __result) {
  sys_time<_Duration> __st{};
  if (!chrono::__from_fields(__f, __st))
    return false;

  __result = file_clock::from_sys(__st);
  return true;
}

#    if _LIBCPP_HAS_EXPERIMENTAL_TZDB && _LIBCPP_HAS_TIME_ZONE_DATABASE && _LIBCPP_HAS_FILESYSTEM
template <class _Duration>
_LIBCPP_HIDE_FROM_ABI bool __from_fields(const __fields_storage& __f, utc_time<_Duration>& __result) {
  if (!chrono::__validate_time_precision<_Duration>(__f))
    return false;

  sys_days __date{};
  if (!__try_get_date(__f, __date))
    return false;

  // utc_time can represent leap seconds, so the seconds field may be 60.
  seconds __time_of_day{};
  if (!__try_get_time_of_day(__f, __time_of_day, 60))
    return false;

  const sys_seconds __sys = __date + __time_of_day - minutes{__f.__utc_offset_};
  utc_seconds __time;
  if (__f.__seconds_ == 60) {
    // Convert the preceding second, then advance into the possible leap second.
    // This also handles offsets that put the leap second on a different local date.
    __time = utc_clock::from_sys(__sys - seconds{1}) + seconds{1};
    if (!chrono::get_leap_second_info(__time).is_leap_second)
      return false;
  } else {
    // Reject the UTC second removed by a negative leap second.
    for (const auto& __leap : chrono::get_tzdb().leap_seconds) {
      if (__sys < __leap.date()) {
        if (__leap.value() < seconds{0} && __sys >= __leap.date() + __leap.value())
          return false;
        break;
      }
    }
    __time = utc_clock::from_sys(__sys);
  }
  using _Precision = common_type_t<_Duration, seconds>;
  __result         = utc_time<_Duration>{chrono::floor<_Duration>(
      chrono::__make_duration<_Precision>(__time.time_since_epoch().count(), __f.__subseconds_))};
  return true;
}

template <class _Duration>
_LIBCPP_HIDE_FROM_ABI bool __from_fields(const __fields_storage& __f, tai_time<_Duration>& __result) {
  if (!chrono::__validate_time_precision<_Duration>(__f))
    return false;

  sys_days __date{};
  if (!__try_get_date(__f, __date))
    return false;

  seconds __time_of_day{};
  if (!__try_get_time_of_day(__f, __time_of_day))
    return false;

  constexpr sys_days __tai_epoch{-days{4383}}; // 1958-01-01.
  const seconds __whole_seconds = __date - __tai_epoch + __time_of_day - minutes{__f.__utc_offset_};
  using _Precision              = common_type_t<_Duration, seconds>;
  __result                      = tai_time<_Duration>{
      chrono::floor<_Duration>(chrono::__make_duration<_Precision>(__whole_seconds.count(), __f.__subseconds_))};
  return true;
}

template <class _Duration>
_LIBCPP_HIDE_FROM_ABI bool __from_fields(const __fields_storage& __f, gps_time<_Duration>& __result) {
  if (!chrono::__validate_time_precision<_Duration>(__f))
    return false;

  sys_days __date{};
  if (!__try_get_date(__f, __date))
    return false;

  seconds __time_of_day{};
  if (!__try_get_time_of_day(__f, __time_of_day))
    return false;

  constexpr sys_days __gps_epoch{days{3657}}; // 1980-01-06.
  const seconds __whole_seconds = __date - __gps_epoch + __time_of_day - minutes{__f.__utc_offset_};
  using _Precision              = common_type_t<_Duration, seconds>;
  __result                      = gps_time<_Duration>{
      chrono::floor<_Duration>(chrono::__make_duration<_Precision>(__whole_seconds.count(), __f.__subseconds_))};
  return true;
}
#    endif // _LIBCPP_HAS_EXPERIMENTAL_TZDB && _LIBCPP_HAS_TIME_ZONE_DATABASE && _LIBCPP_HAS_FILESYSTEM

// Calendrical results reject fields they cannot represent. UTC offsets are
// excluded from these checks; time zone abbreviations are stored separately.
_LIBCPP_HIDE_FROM_ABI inline bool __from_fields(const __fields_storage& __f, day& __result) {
  if (!__f.__has_exactly(__fields_set::__day))
    return false;

  if (!__in_range(__f.__day_, 1, 31))
    return false;

  __result = day{static_cast<unsigned>(__f.__day_)};
  return true;
}

_LIBCPP_HIDE_FROM_ABI inline bool __from_fields(const __fields_storage& __f, month& __result) {
  if (!__f.__has_exactly(__fields_set::__month))
    return false;

  if (!__in_range(__f.__month_, 1, 12))
    return false;

  __result = month{static_cast<unsigned>(__f.__month_)};
  return true;
}

_LIBCPP_HIDE_FROM_ABI inline bool __from_fields(const __fields_storage& __f, year& __result) {
  constexpr auto __year_fields = __fields_set::__year | __fields_set::__century | __fields_set::__year_of_century;
  if (!__f.__has_only(__year_fields))
    return false;

  int __year{};
  if (!__try_get_year(__f, __year))
    return false;

  __result = year{__year};
  return true;
}

_LIBCPP_HIDE_FROM_ABI inline bool __from_fields(const __fields_storage& __f, weekday& __result) {
  if (!__f.__has_exactly(__fields_set::__weekday))
    return false;

  if (!__in_range(__f.__weekday_, 0, 6))
    return false;

  __result = weekday{static_cast<unsigned>(__f.__weekday_)};
  return true;
}

_LIBCPP_HIDE_FROM_ABI inline bool __from_fields(const __fields_storage& __f, month_day& __result) {
  if (!__f.__has_only(__fields_set::__month | __fields_set::__day | __fields_set::__day_of_year))
    return false;

  int __month{};
  int __day{};
  if (__f.__has(__fields_set::__day_of_year)) {
    // Without a year, only days through February 28 are unambiguous.
    if (!__in_range(__f.__day_of_year_, 1, 59))
      return false;
    __month = __f.__day_of_year_ <= 31 ? 1 : 2;
    __day   = __f.__day_of_year_ <= 31 ? __f.__day_of_year_ : __f.__day_of_year_ - 31;
    if ((__f.__has(__fields_set::__month) && __f.__month_ != __month) ||
        (__f.__has(__fields_set::__day) && __f.__day_ != __day))
      return false;
  } else {
    if (!__f.__has(__fields_set::__month | __fields_set::__day))
      return false;
    __month = __f.__month_;
    __day   = __f.__day_;
  }

  if (!__in_range(__month, 1, 12) || !__in_range(__day, 1, 31))
    return false;

  month_day __md{month{static_cast<unsigned>(__month)}, day{static_cast<unsigned>(__day)}};
  if (!__md.ok())
    return false;

  __result = __md;
  return true;
}

_LIBCPP_HIDE_FROM_ABI inline bool __from_fields(const __fields_storage& __f, year_month& __result) {
  constexpr auto __year_fields = __fields_set::__year | __fields_set::__century | __fields_set::__year_of_century;
  if (!__f.__has_exactly(__fields_set::__month, __year_fields))
    return false;

  int __year{};
  if (!__try_get_year(__f, __year))
    return false;

  if (!__in_range(__f.__month_, 1, 12))
    return false;

  __result = year_month{year{__year}, month{static_cast<unsigned>(__f.__month_)}};
  return true;
}

_LIBCPP_HIDE_FROM_ABI inline bool __from_fields(const __fields_storage& __f, year_month_day& __result) {
  constexpr auto __date_fields =
      __fields_set::__year | __fields_set::__century | __fields_set::__year_of_century | __fields_set::__month |
      __fields_set::__day | __fields_set::__iso_year | __fields_set::__iso_year_of_century | __fields_set::__iso_week |
      __fields_set::__weekday | __fields_set::__day_of_year | __fields_set::__week_sun | __fields_set::__week_mon;
  if (!__f.__has_only(__date_fields))
    return false;

  // Resolve the date and check that all supplied date fields agree.
  sys_days __date{};
  if (!__try_get_date(__f, __date))
    return false;

  __result = year_month_day{__date};
  return true;
}

template <class _Tp, class _CharT, class _Traits, class _Alloc>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
__from_stream(basic_istream<_CharT, _Traits>& __is,
              const _CharT* __fmt,
              _Tp& __value,
              basic_string<_CharT, _Traits, _Alloc>* __abbrev,
              minutes* __offset) {
  ios_base::iostate __state = ios_base::goodbit;
  constexpr bool __noskipws = true;
  typename basic_istream<_CharT, _Traits>::sentry __s{__is, __noskipws};

  if (__s) {
#    if _LIBCPP_HAS_EXCEPTIONS
    try {
#    endif
      __fields_storage __f{};
      basic_string<_CharT, _Traits> __parsed_abbrev;
      constexpr auto __options = chrono::__get_parse_options<_Tp>();

      // Parse the input according to the format and collect the fields.
      __from_stream_parser<_CharT, _Traits> __parser{__is, __state};
      __parser.__parse(__fmt, __f, __parsed_abbrev, __options);
      if (!__parser.__fail()) {
        if (__abbrev && !__parsed_abbrev.empty())
          __abbrev->assign(__parsed_abbrev.data(), __parsed_abbrev.size());
        if (__offset && __f.__has(__fields_set::__utc_offset))
          *__offset = minutes{__f.__utc_offset_};

        // Resolve and validate the fields needed to construct the requested result.
        _Tp __result{};
        if (!chrono::__from_fields(__f, __result)) {
          __state |= ios_base::failbit;
        } else {
          __value = __result;
        }
      }
#    if _LIBCPP_HAS_EXCEPTIONS
    } catch (...) {
      __state |= ios_base::badbit;
      __is.__setstate_nothrow(__state);
      if (__is.exceptions() & ios_base::badbit)
        throw;
    }
#    endif
    // Keep exceptions requested for failbit or eofbit outside the input handler.
    __is.setstate(__state);
  }

  return __is;
}

template <class _CharT, class _Traits, class _Rep, class _Period, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            duration<_Rep, _Period>& __d,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __d, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Duration, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            sys_time<_Duration>& __tp,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __tp, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Duration, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            local_time<_Duration>& __tp,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __tp, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Duration, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            file_time<_Duration>& __tp,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __tp, __abbrev, __offset);
}

#    if _LIBCPP_HAS_EXPERIMENTAL_TZDB && _LIBCPP_HAS_TIME_ZONE_DATABASE && _LIBCPP_HAS_FILESYSTEM
template <class _CharT, class _Traits, class _Duration, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            utc_time<_Duration>& __tp,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __tp, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Duration, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            tai_time<_Duration>& __tp,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __tp, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Duration, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            gps_time<_Duration>& __tp,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __tp, __abbrev, __offset);
}
#    endif // _LIBCPP_HAS_EXPERIMENTAL_TZDB && _LIBCPP_HAS_TIME_ZONE_DATABASE && _LIBCPP_HAS_FILESYSTEM

template <class _CharT, class _Traits, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            day& __d,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __d, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            month& __m,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __m, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            year& __y,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __y, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            weekday& __wd,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __wd, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            month_day& __md,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __md, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            year_month& __ym,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __ym, __abbrev, __offset);
}

template <class _CharT, class _Traits, class _Alloc = allocator<_CharT>>
_LIBCPP_HIDE_FROM_ABI basic_istream<_CharT, _Traits>&
from_stream(basic_istream<_CharT, _Traits>& __is,
            const _CharT* __fmt,
            year_month_day& __ymd,
            basic_string<_CharT, _Traits, _Alloc>* __abbrev = nullptr,
            minutes* __offset                               = nullptr) {
  return chrono::__from_stream(__is, __fmt, __ymd, __abbrev, __offset);
}

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD

#  endif

_LIBCPP_POP_MACROS

#endif

#endif //_LIBCPP___CHRONO_FROM_STREAM_H
