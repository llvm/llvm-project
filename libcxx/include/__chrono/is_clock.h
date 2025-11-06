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

#include <__chrono/duration.h>
#include <__chrono/time_point.h>
#include <__concepts/same_as.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_class.h>
#include <__type_traits/is_union.h>
#include <ratio>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

// Helper to check that _Tp::time_point has the form time_point<_, typename _Tp::duration>.
template <class _TimePoint, class _ClockType>
inline constexpr bool __is_valid_clock_time_point_v = false;

template <class _TimePointClock, class _ClockType>
inline constexpr bool
    __is_valid_clock_time_point_v<time_point<_TimePointClock, typename _ClockType::duration>, _ClockType> = true;

// Check if a clock satisfies the Cpp17Clock requirements as defined in [time.clock.req]
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_clock_v = requires {
  typename _Tp::rep;
  requires is_arithmetic_v<typename _Tp::rep> || is_class_v<typename _Tp::rep> || is_union_v<typename _Tp::rep>;

  typename _Tp::period;
  requires __is_ratio_v<typename _Tp::period>;

  typename _Tp::duration;
  requires same_as<typename _Tp::duration, duration<typename _Tp::rep, typename _Tp::period>>;

  typename _Tp::time_point;
  requires __is_valid_clock_time_point_v<typename _Tp::time_point, _Tp>;

  _Tp::is_steady;
  requires same_as<decltype((_Tp::is_steady)), const bool&>;

  _Tp::now();
  requires same_as<decltype(_Tp::now()), typename _Tp::time_point>;
};

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_clock : bool_constant<is_clock_v<_Tp>> {};

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER
#endif // _LIBCPP___CHRONO_IS_CLOCK_H
