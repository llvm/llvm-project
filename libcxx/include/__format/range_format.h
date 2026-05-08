// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_RANGE_FORMAT_H
#define _LIBCPP___FORMAT_RANGE_FORMAT_H

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#include <__concepts/same_as.h>
#include <__config>
#include <__format/fmt_pair_like.h>
#include <__fwd/format.h>
#include <__ranges/concepts.h>
#include <__type_traits/remove_cvref.h>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

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

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif
