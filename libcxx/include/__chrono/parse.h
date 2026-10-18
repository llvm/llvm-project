// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_PARSE_H
#define _LIBCPP___CHRONO_PARSE_H

#include <__config>

#if _LIBCPP_HAS_LOCALIZATION

#  include <__chrono/duration.h>
#  include <__chrono/from_stream.h>
#  include <__fwd/istream.h>
#  include <__fwd/memory.h>
#  include <__fwd/string.h>
#  include <__memory/addressof.h>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

#  if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

namespace __parse_details {

// Block ordinary lookup so from_stream is found only through ADL.
void from_stream() = delete;

// parse() stores the format and output references in a manipulator.
// operator>> supplies the input stream and calls the matching from_stream overload.
// Separate manipulator types preserve the four from_stream argument lists.
// _Format preserves the value category of fmt (lvalue) or fmt.c_str() (prvalue).

template <class _CharT, class _Traits, class _Parsable, class _Format>
struct __parse_manip {
  const _CharT* __fmt_;
  _Parsable* __tp_;

  _LIBCPP_HIDE_FROM_ABI __parse_manip(const _CharT* __fmt, _Parsable* __tp) : __fmt_(__fmt), __tp_(__tp) {}

  __parse_manip(const __parse_manip&)            = delete;
  __parse_manip& operator=(const __parse_manip&) = delete;

  _LIBCPP_HIDE_FROM_ABI friend basic_istream<_CharT, _Traits>&
  operator>>(basic_istream<_CharT, _Traits>& __is, __parse_manip&& __manip) {
    // NOLINTNEXTLINE(libcpp-robust-against-adl) [time.parse] requires ADL.
    from_stream(__is, static_cast<_Format>(__manip.__fmt_), *__manip.__tp_);
    return __is;
  }
};

template <class _CharT, class _Traits, class _Parsable, class _Format, class _Alloc = allocator<_CharT>>
struct __parse_manip_offset {
  const _CharT* __fmt_;
  _Parsable* __tp_;
  minutes* __offset_;

  _LIBCPP_HIDE_FROM_ABI __parse_manip_offset(const _CharT* __fmt, _Parsable* __tp, minutes* __offset)
      : __fmt_(__fmt), __tp_(__tp), __offset_(__offset) {}

  __parse_manip_offset(const __parse_manip_offset&)            = delete;
  __parse_manip_offset& operator=(const __parse_manip_offset&) = delete;

  _LIBCPP_HIDE_FROM_ABI friend basic_istream<_CharT, _Traits>&
  operator>>(basic_istream<_CharT, _Traits>& __is, __parse_manip_offset&& __manip) {
    // NOLINTNEXTLINE(libcpp-robust-against-adl) [time.parse] requires ADL.
    from_stream(__is,
                static_cast<_Format>(__manip.__fmt_),
                *__manip.__tp_,
                static_cast<basic_string<_CharT, _Traits, _Alloc>*>(nullptr),
                static_cast<minutes*>(__manip.__offset_));
    return __is;
  }
};

template <class _CharT, class _Traits, class _Alloc, class _Parsable, class _Format>
struct __parse_manip_abbrev {
  const _CharT* __fmt_;
  _Parsable* __tp_;
  basic_string<_CharT, _Traits, _Alloc>* __abbrev_;

  _LIBCPP_HIDE_FROM_ABI
  __parse_manip_abbrev(const _CharT* __fmt, _Parsable* __tp, basic_string<_CharT, _Traits, _Alloc>* __abbrev)
      : __fmt_(__fmt), __tp_(__tp), __abbrev_(__abbrev) {}

  __parse_manip_abbrev(const __parse_manip_abbrev&)            = delete;
  __parse_manip_abbrev& operator=(const __parse_manip_abbrev&) = delete;

  _LIBCPP_HIDE_FROM_ABI friend basic_istream<_CharT, _Traits>&
  operator>>(basic_istream<_CharT, _Traits>& __is, __parse_manip_abbrev&& __manip) {
    // NOLINTNEXTLINE(libcpp-robust-against-adl) [time.parse] requires ADL.
    from_stream(__is,
                static_cast<_Format>(__manip.__fmt_),
                *__manip.__tp_,
                static_cast<basic_string<_CharT, _Traits, _Alloc>*>(__manip.__abbrev_));
    return __is;
  }
};

template <class _CharT, class _Traits, class _Alloc, class _Parsable, class _Format>
struct __parse_manip_abbrev_offset {
  const _CharT* __fmt_;
  _Parsable* __tp_;
  basic_string<_CharT, _Traits, _Alloc>* __abbrev_;
  minutes* __offset_;

  _LIBCPP_HIDE_FROM_ABI __parse_manip_abbrev_offset(
      const _CharT* __fmt, _Parsable* __tp, basic_string<_CharT, _Traits, _Alloc>* __abbrev, minutes* __offset)
      : __fmt_(__fmt), __tp_(__tp), __abbrev_(__abbrev), __offset_(__offset) {}

  __parse_manip_abbrev_offset(const __parse_manip_abbrev_offset&)            = delete;
  __parse_manip_abbrev_offset& operator=(const __parse_manip_abbrev_offset&) = delete;

  _LIBCPP_HIDE_FROM_ABI friend basic_istream<_CharT, _Traits>&
  operator>>(basic_istream<_CharT, _Traits>& __is, __parse_manip_abbrev_offset&& __manip) {
    // NOLINTNEXTLINE(libcpp-robust-against-adl) [time.parse] requires ADL.
    from_stream(__is,
                static_cast<_Format>(__manip.__fmt_),
                *__manip.__tp_,
                static_cast<basic_string<_CharT, _Traits, _Alloc>*>(__manip.__abbrev_),
                static_cast<minutes*>(__manip.__offset_));
    return __is;
  }
};

// [time.parse]: a Parsable is anything from_stream can read, with the trailing
// arguments the selected parse overload passes on. The call is unqualified, so
// a user-defined type that provides its own from_stream is parsable as well.
// _Format and _Args model the types and value categories of the format and trailing arguments.
template <class _Parsable, class _CharT, class _Traits, class _Format, class... _Args>
concept __parsable =
    requires(basic_istream<_CharT, _Traits>& __is, const _CharT* __fmt, _Parsable& __tp, _Args... __args) {
      // NOLINTNEXTLINE(libcpp-robust-against-adl) [time.parse] requires ADL.
      from_stream(__is, static_cast<_Format>(__fmt), __tp, static_cast<_Args>(__args)...);
    };

} // namespace __parse_details

template <class _CharT, class _Parsable>
  requires __parse_details::__parsable<_Parsable, _CharT, char_traits<_CharT>, const _CharT*&>
_LIBCPP_HIDE_FROM_ABI __parse_details::__parse_manip<_CharT, char_traits<_CharT>, _Parsable, const _CharT*&>
parse(const _CharT* __fmt, _Parsable& __tp) {
  return {__fmt, std::addressof(__tp)};
}

template <class _CharT, class _Traits, class _Alloc, class _Parsable>
  requires __parse_details::__parsable<_Parsable, _CharT, _Traits, const _CharT*>
_LIBCPP_HIDE_FROM_ABI __parse_details::__parse_manip<_CharT, _Traits, _Parsable, const _CharT*>
parse(const basic_string<_CharT, _Traits, _Alloc>& __fmt, _Parsable& __tp) {
  return {__fmt.c_str(), std::addressof(__tp)};
}

template <class _CharT, class _Traits, class _Alloc, class _Parsable>
  requires __parse_details::
      __parsable<_Parsable, _CharT, _Traits, const _CharT*&, basic_string<_CharT, _Traits, _Alloc>*>
    _LIBCPP_HIDE_FROM_ABI __parse_details::__parse_manip_abbrev<_CharT, _Traits, _Alloc, _Parsable, const _CharT*&>
    parse(const _CharT* __fmt, _Parsable& __tp, basic_string<_CharT, _Traits, _Alloc>& __abbrev) {
  return {__fmt, std::addressof(__tp), std::addressof(__abbrev)};
}

template <class _CharT, class _Traits, class _Alloc, class _Parsable>
  requires __parse_details::
      __parsable<_Parsable, _CharT, _Traits, const _CharT*, basic_string<_CharT, _Traits, _Alloc>*>
    _LIBCPP_HIDE_FROM_ABI __parse_details::__parse_manip_abbrev<_CharT, _Traits, _Alloc, _Parsable, const _CharT*>
    parse(const basic_string<_CharT, _Traits, _Alloc>& __fmt,
          _Parsable& __tp,
          basic_string<_CharT, _Traits, _Alloc>& __abbrev) {
  return {__fmt.c_str(), std::addressof(__tp), std::addressof(__abbrev)};
}

template <class _CharT, class _Parsable>
  requires __parse_details::
      __parsable<_Parsable, _CharT, char_traits<_CharT>, const _CharT*&, basic_string<_CharT>*, minutes*>
    _LIBCPP_HIDE_FROM_ABI __parse_details::__parse_manip_offset<_CharT, char_traits<_CharT>, _Parsable, const _CharT*&>
    parse(const _CharT* __fmt, _Parsable& __tp, minutes& __offset) {
  return {__fmt, std::addressof(__tp), std::addressof(__offset)};
}

template <class _CharT, class _Traits, class _Alloc, class _Parsable>
  requires __parse_details::
      __parsable<_Parsable, _CharT, _Traits, const _CharT*, basic_string<_CharT, _Traits, _Alloc>*, minutes*>
    _LIBCPP_HIDE_FROM_ABI __parse_details::__parse_manip_offset<_CharT, _Traits, _Parsable, const _CharT*, _Alloc>
    parse(const basic_string<_CharT, _Traits, _Alloc>& __fmt, _Parsable& __tp, minutes& __offset) {
  return {__fmt.c_str(), std::addressof(__tp), std::addressof(__offset)};
}

template <class _CharT, class _Traits, class _Alloc, class _Parsable>
  requires __parse_details::
      __parsable<_Parsable, _CharT, _Traits, const _CharT*&, basic_string<_CharT, _Traits, _Alloc>*, minutes*>
    _LIBCPP_HIDE_FROM_ABI
    __parse_details::__parse_manip_abbrev_offset<_CharT, _Traits, _Alloc, _Parsable, const _CharT*&>
    parse(const _CharT* __fmt, _Parsable& __tp, basic_string<_CharT, _Traits, _Alloc>& __abbrev, minutes& __offset) {
  return {__fmt, std::addressof(__tp), std::addressof(__abbrev), std::addressof(__offset)};
}

template <class _CharT, class _Traits, class _Alloc, class _Parsable>
  requires __parse_details::
      __parsable<_Parsable, _CharT, _Traits, const _CharT*, basic_string<_CharT, _Traits, _Alloc>*, minutes*>
    _LIBCPP_HIDE_FROM_ABI
    __parse_details::__parse_manip_abbrev_offset<_CharT, _Traits, _Alloc, _Parsable, const _CharT*>
    parse(const basic_string<_CharT, _Traits, _Alloc>& __fmt,
          _Parsable& __tp,
          basic_string<_CharT, _Traits, _Alloc>& __abbrev,
          minutes& __offset) {
  return {__fmt.c_str(), std::addressof(__tp), std::addressof(__abbrev), std::addressof(__offset)};
}

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD

#  endif

#endif

#endif //_LIBCPP___CHRONO_PARSE_H
