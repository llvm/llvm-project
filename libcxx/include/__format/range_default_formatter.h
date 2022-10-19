// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_RANGE_DEFAULT_FORMATTER_H
#define _LIBCPP___FORMAT_RANGE_DEFAULT_FORMATTER_H

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#include <__availability>
#include <__concepts/same_as.h>
#include <__config>
#include <__format/concepts.h>
#include <__format/formatter.h>
#include <__ranges/concepts.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/pair.h>
#include <tuple>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 20

template <class _Rp, class _CharT>
concept __const_formattable_range =
    ranges::input_range<const _Rp> && formattable<ranges::range_reference_t<const _Rp>, _CharT>;

template <class _Rp, class _CharT>
using __fmt_maybe_const = conditional_t<__const_formattable_range<_Rp, _CharT>, const _Rp, _Rp>;

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wshadow")
_LIBCPP_GCC_DIAGNOSTIC_IGNORED("-Wshadow")
// This shadows map, set, and string.
enum class range_format { disabled, map, set, sequence, string, debug_string };
_LIBCPP_DIAGNOSTIC_POP

// There is no definition of this struct, it's purely intended to be used to
// generate diagnostics.
template <class _Rp>
struct _LIBCPP_TEMPLATE_VIS __instantiated_the_primary_template_of_format_kind;

template <class _Rp>
constexpr range_format format_kind = [] {
  // [format.range.fmtkind]/1
  // A program that instantiates the primary template of format_kind is ill-formed.
  static_assert(sizeof(_Rp) != sizeof(_Rp), "create a template specialization of format_kind for your type");
  return range_format::disabled;
}();

template <ranges::input_range _Rp>
  requires same_as<_Rp, remove_cvref_t<_Rp>>
inline constexpr range_format format_kind<_Rp> = [] {
  // [format.range.fmtkind]/2

  // 2.1 If same_as<remove_cvref_t<ranges::range_reference_t<R>>, R> is true,
  // Otherwise format_kind<R> is range_format::disabled.
  if constexpr (same_as<remove_cvref_t<ranges::range_reference_t<_Rp>>, _Rp>)
    return range_format::disabled;
  // 2.2 Otherwise, if the qualified-id R::key_type is valid and denotes a type:
  else if constexpr (requires { typename _Rp::key_type; }) {
    // 2.2.1 If the qualified-id R::mapped_type is valid and denotes a type ...
    if constexpr (requires { typename _Rp::mapped_type; } &&
                  // 2.2.1 ... If either U is a specialization of pair or U is a specialization
                  // of tuple and tuple_size_v<U> == 2
                  __fmt_pair_like<remove_cvref_t<ranges::range_reference_t<_Rp>>>)
      return range_format::map;
    else
      // 2.2.2 Otherwise format_kind<R> is range_format::set.
      return range_format::set;
  } else
    // 2.3 Otherwise, format_kind<R> is range_format::sequence.
    return range_format::sequence;
}();

// This is a non-standard work-around to fix instantiation of
//   formatter<const _CharT[N], _CharT>
// const _CharT[N] satisfies the ranges::input_range concept.
// remove_cvref_t<const _CharT[N]> is _CharT[N] so it does not satisfy the
// requirement of the above specialization. Instead it will instantiate the
// primary template, which is ill-formed.
//
// An alternative solution is to remove the offending formatter.
//
// https://godbolt.org/z/bqjhhaexx
//
// The removal is proposed in LWG3833, but use the work-around until the issue
// has been adopted.
// TODO FMT Implement LWG3833.
template <class _CharT, size_t N>
inline constexpr range_format format_kind<const _CharT[N]> = range_format::disabled;

template <range_format _Kp, ranges::input_range _Rp, class _CharT>
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT __range_default_formatter;

// Required specializations

template <ranges::input_range _Rp, class _CharT>
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT __range_default_formatter<range_format::sequence, _Rp, _CharT> {
  __range_default_formatter() = delete; // TODO FMT Implement
};

template <ranges::input_range _Rp, class _CharT>
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT __range_default_formatter<range_format::map, _Rp, _CharT> {
  __range_default_formatter() = delete; // TODO FMT Implement
};

template <ranges::input_range _Rp, class _CharT>
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT __range_default_formatter<range_format::set, _Rp, _CharT> {
  __range_default_formatter() = delete; // TODO FMT Implement
};

template <range_format _Kp, ranges::input_range _Rp, class _CharT>
  requires(_Kp == range_format::string || _Kp == range_format::debug_string)
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT __range_default_formatter<_Kp, _Rp, _CharT> {
  __range_default_formatter() = delete; // TODO FMT Implement
};

// Dispatcher to select the specialization based on the type of the range.

template <ranges::input_range _Rp, class _CharT>
  requires(format_kind<_Rp> != range_format::disabled && formattable<ranges::range_reference_t<_Rp>, _CharT>)
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT formatter<_Rp, _CharT>
    : __range_default_formatter<format_kind<_Rp>, _Rp, _CharT> {};

#endif //_LIBCPP_STD_VER > 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FORMAT_RANGE_DEFAULT_FORMATTER_H
