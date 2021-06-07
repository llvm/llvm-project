// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_VIEW_H
#define _LIBCPP___RANGES_VIEW_H

#include <__config>
#include <__ranges/concepts.h>
#include <concepts>


#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_RANGES)

namespace ranges {

struct view_base { };

template <class _Tp>
inline constexpr bool enable_view = derived_from<_Tp, view_base>;

template <class _Tp>
concept view =
  range<_Tp> &&
  movable<_Tp> &&
  default_initializable<_Tp> &&
  enable_view<_Tp>;

template<class _Range>
concept __simple_view =
  view<_Range> && range<const _Range> &&
  same_as<iterator_t<_Range>, iterator_t<const _Range>> &&
  same_as<sentinel_t<_Range>, iterator_t<const _Range>>;
} // end namespace ranges

#endif // !_LIBCPP_HAS_NO_RANGES

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_VIEW_H
