// -*- C++ -*-
//===--------------------- __ranges/concepts.h ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP_RANGES_CONCEPTS_H
#define _LIBCPP_RANGES_CONCEPTS_H

#include <__config>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// clang-format off

#if !defined(_LIBCPP_HAS_NO_RANGES)

namespace ranges {
  // [range.range]
  template <class _Tp>
  concept range = requires(_Tp& __t) {
    ranges::begin(__t); // sometimes equality-preserving
    ranges::end(__t);
  };

  // `iterator_t` defined in <__ranges/access.h>

  template <range _Rp>
  using sentinel_t = decltype(ranges::end(declval<_Rp&>()));

  template <range _Rp>
  using range_difference_t = iter_difference_t<iterator_t<_Rp> >;

  template <range _Rp>
  using range_value_t = iter_value_t<iterator_t<_Rp> >;

  template <range _Rp>
  using range_reference_t = iter_reference_t<iterator_t<_Rp> >;

  template <range _Rp>
  using range_rvalue_reference_t = iter_rvalue_reference_t<iterator_t<_Rp> >;

  // [range.refinements], other range refinements
  template <class _Tp>
  concept input_range = range<_Tp> && input_iterator<iterator_t<_Tp> >;

  template <class _Tp>
  concept forward_range = input_range<_Tp> && forward_iterator<iterator_t<_Tp> >;

  template <class _Tp>
  concept bidirectional_range = forward_range<_Tp> && bidirectional_iterator<iterator_t<_Tp> >;

  template <class _Tp>
  concept common_range = range<_Tp> && same_as<iterator_t<_Tp>, sentinel_t<_Tp> >;

  template <class _Tp>
  concept random_access_range =
      bidirectional_range<_Tp> && random_access_iterator<iterator_t<_Tp> >;
} // namespace ranges

#endif // !defined(_LIBCPP_HAS_NO_RANGES)

// clang-format on

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP_RANGES_CONCEPTS_H
