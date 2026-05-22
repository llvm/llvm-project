// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_OPTIONAL_SWAP_H
#define _LIBCPP_OPTIONAL_SWAP_H

#include <__config>
#include <__fwd/optional.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_swappable.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, enable_if_t< is_move_constructible_v<_Tp> && is_swappable_v<_Tp>, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
swap(optional<_Tp>& __x, optional<_Tp>& __y) noexcept(noexcept(__x.swap(__y))) {
  __x.swap(__y);
}

_LIBCPP_END_NAMESPACE_STD

#endif

#endif
