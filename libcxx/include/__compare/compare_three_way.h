// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___COMPARE_COMPARE_THREE_WAY_H
#define _LIBCPP___COMPARE_COMPARE_THREE_WAY_H

#include <__compare/three_way_comparable.h>
#include <__config>
#include <__type_traits/desugars_to.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

struct compare_three_way {
  template <class _T1, class _T2>
    requires three_way_comparable_with<_T1, _T2>
  constexpr _LIBCPP_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
      noexcept(noexcept(std::forward<_T1>(__t) <=> std::forward<_T2>(__u))) {
    return std::forward<_T1>(__t) <=> std::forward<_T2>(__u);
  }

  using is_transparent = void;
};

template <class _Tp, class _Up>
inline const bool __desugars_to_v<__compare_three_way_tag, compare_three_way, _Tp, _Up> = true;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___COMPARE_COMPARE_THREE_WAY_H
