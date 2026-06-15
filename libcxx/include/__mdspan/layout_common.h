// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___MDSPAN_LAYOUT_COMMON_H
#define _LIBCPP___MDSPAN_LAYOUT_COMMON_H

#include <__concepts/same_as.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__fwd/mdspan.h>
#include <__fwd/span.h>
#include <__mdspan/extents.h>
#include <__memory/addressof.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_same.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace __mdspan_detail {

template <class _Layout, class _Mapping>
constexpr bool __is_mapping_of =
    is_same_v<typename _Layout::template mapping<typename _Mapping::extents_type>, _Mapping>;

template <class _Mapping>
concept __layout_mapping_alike = requires {
  requires __is_mapping_of<typename _Mapping::layout_type, _Mapping>;
  requires __is_extents_v<typename _Mapping::extents_type>;
  { _Mapping::is_always_strided() } -> same_as<bool>;
  { _Mapping::is_always_exhaustive() } -> same_as<bool>;
  { _Mapping::is_always_unique() } -> same_as<bool>;
  bool_constant<_Mapping::is_always_strided()>::value;
  bool_constant<_Mapping::is_always_exhaustive()>::value;
  bool_constant<_Mapping::is_always_unique()>::value;
};

#  if _LIBCPP_STD_VER >= 26

template <class _Integral>
_LIBCPP_HIDE_FROM_ABI constexpr _Integral __least_multiple_at_least(_Integral __multiplier, _Integral __minimum) {
  if (__multiplier == static_cast<_Integral>(0))
    return __minimum;

  _Integral __factor = __minimum / __multiplier;
  if (__minimum % __multiplier != static_cast<_Integral>(0))
    ++__factor;

  return __factor * __multiplier;
}

template <class _To, class _Integral>
_LIBCPP_HIDE_FROM_ABI constexpr bool
__least_multiple_at_least_is_representable_as(_Integral __multiplier, _Integral __minimum) {
  if (__multiplier == static_cast<_Integral>(0))
    return __mdspan_detail::__is_representable_as<_To>(__minimum);

  _Integral __factor = __minimum / __multiplier;
  if (__minimum % __multiplier != static_cast<_Integral>(0)) {
    bool __overflowed_add = __builtin_add_overflow(__factor, static_cast<_Integral>(1), std::addressof(__factor));
    if (__overflowed_add)
      return false;
  }

  _Integral __result    = 0;
  bool __overflowed_mul = __builtin_mul_overflow(__factor, __multiplier, std::addressof(__result));
  return !__overflowed_mul && __mdspan_detail::__is_representable_as<_To>(__result);
}

template <template <size_t> class _LayoutTemplate, class _Layout>
struct __is_layout_specialization_of : false_type {};

template <template <size_t> class _LayoutTemplate, size_t _PaddingValue>
struct __is_layout_specialization_of<_LayoutTemplate, _LayoutTemplate<_PaddingValue>> : true_type {};

template <class _Mapping>
concept __layout_left_padded_mapping_of =
    __layout_mapping_alike<_Mapping> &&
    __is_layout_specialization_of<layout_left_padded, typename _Mapping::layout_type>::value;

template <class _Mapping>
concept __layout_right_padded_mapping_of =
    __layout_mapping_alike<_Mapping> &&
    __is_layout_specialization_of<layout_right_padded, typename _Mapping::layout_type>::value;

template <class _Mapping>
concept __layout_right_mapping_of = __layout_mapping_alike<_Mapping> && __is_mapping_of<layout_right, _Mapping>;

template <class _Mapping>
concept __layout_left_mapping_of = __layout_mapping_alike<_Mapping> && __is_mapping_of<layout_left, _Mapping>;

_LIBCPP_HIDE_FROM_ABI constexpr size_t
__compute_static_padding_stride(size_t __rank, size_t __padding_value, size_t __static_extent) {
  if (__rank <= 1)
    return 0uz;
  if (__padding_value == dynamic_extent || __static_extent == dynamic_extent)
    return dynamic_extent;
  return __least_multiple_at_least(__padding_value, __static_extent);
}

template <class _Mapping>
  requires __layout_left_padded_mapping_of<_Mapping> || __layout_right_padded_mapping_of<_Mapping>
constexpr size_t __static_padding_stride_of = [] {
  using _Extents          = _Mapping::extents_type;
  constexpr size_t __rank = _Extents::rank();

  if constexpr (__layout_left_padded_mapping_of<_Mapping>) {
    constexpr size_t __static_extent = __rank == 0 ? 0 : _Extents::static_extent(0);
    return __compute_static_padding_stride(__rank, _Mapping::padding_value, __static_extent);
  }
  if constexpr (__layout_right_padded_mapping_of<_Mapping>) {
    constexpr size_t __static_extent = __rank == 0 ? 0 : _Extents::static_extent(__rank - 1);
    return __compute_static_padding_stride(__rank, _Mapping::padding_value, __static_extent);
  }
}();

#  endif // _LIBCPP_STD_VER >= 26

} // namespace __mdspan_detail

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MDSPAN_LAYOUT_PADDED_COMMON_H
