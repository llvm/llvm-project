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
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_same.h>
#include <ratio>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

template <class _Rep, class _Period>
class duration;

template <class _Clock, class _Duration>
class time_point;

// Check if a clock satisfies the Cpp17Clock requirements as defined in [time.clock.req]
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_clock_v = requires {
  typename _Tp::rep;
  requires is_arithmetic_v<typename _Tp::rep>;

  typename _Tp::period;
  requires __is_ratio_v<typename _Tp::period>;

  typename _Tp::duration;
  requires _IsSame<typename _Tp::duration, duration<typename _Tp::rep, typename _Tp::period>>::value;

  typename _Tp::time_point;
  requires _IsSame<typename _Tp::time_point::duration, typename _Tp::duration>::value;

  _Tp::is_steady;
  requires _IsSame<decltype(_Tp::is_steady), const bool>::value;

  _Tp::now();
  requires _IsSame<decltype(_Tp::now()), typename _Tp::time_point>::value;
};

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_clock : bool_constant<is_clock_v<_Tp>> {};

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER
#endif // _LIBCPP___CHRONO_IS_CLOCK_H
