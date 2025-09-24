// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_IS_CLOCK_H
#define _LIBCPP___CHRONO_IS_CLOCK_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

#  include <__type_traits/integral_constant.h>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

template <class>
struct is_clock : std::false_type {};

template <class _Tp>
  requires requires {
    typename _Tp::rep;
    typename _Tp::period;
    typename _Tp::duration;
    typename _Tp::time_point;
    _Tp::is_steady;
    _Tp::now();
  }
struct is_clock<_Tp> : std::true_type {};

template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_clock_v = is_clock<_Tp>::value;

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER
#endif // _LIBCPP___CHRONO_IS_CLOCK_H
