// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_CLOCK_CAST_H
#define _LIBCPP___CHRONO_CLOCK_CAST_H

#include <__config>

#include <__chrono/duration.h>
#include <__chrono/system_clock.h>
#include <__chrono/time_point.h>
#include <__type_traits/common_type.h>

#if _LIBCPP_HAS_TIME_ZONE_DATABASE && _LIBCPP_HAS_FILESYSTEM && _LIBCPP_HAS_LOCALIZATION
#  include <__chrono/utc_clock.h>
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

template <class _Tp, class _Clock>
inline constexpr bool __is_time_point_of_clock_v = false;

template <class _Clock, class _Duration>
inline constexpr bool __is_time_point_of_clock_v<time_point<_Clock, _Duration>, _Clock> = true;

// [time.clock.conv]
template <class _DestClock, class _SourceClock>
struct clock_time_conversion {};

// [time.clock.cast.id]
template <class _Clock>
struct clock_time_conversion<_Clock, _Clock> {
  template <class _Duration>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI time_point<_Clock, _Duration>
  operator()(const time_point<_Clock, _Duration>& __t) const {
    return __t;
  }
};

template <>
struct clock_time_conversion<system_clock, system_clock> {
  template <class _Duration>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI sys_time<_Duration> operator()(const sys_time<_Duration>& __t) const {
    return __t;
  }
};

// [time.clock.cast.sys]
//
// The trailing return types implement the Constraints: elements and the
// static_asserts implement the Mandates: elements. The clocks are named as
// time_point<_Clock, _Duration>::clock in the return types, so that the lookup
// of to_sys etc. depends on _Duration. Otherwise it would happen when the class
// is instantiated, resulting in a hard error for clocks without these members.
template <class _SourceClock>
struct clock_time_conversion<system_clock, _SourceClock> {
  template <class _Duration>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI auto operator()(const time_point<_SourceClock, _Duration>& __t) const
      -> decltype(time_point<_SourceClock, _Duration>::clock::to_sys(__t)) {
    static_assert(__is_time_point_of_clock_v<decltype(_SourceClock::to_sys(__t)), system_clock>,
                  "SourceClock::to_sys(t) must return a sys_time");
    return _SourceClock::to_sys(__t);
  }
};

template <class _DestClock>
struct clock_time_conversion<_DestClock, system_clock> {
  template <class _Duration>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI auto operator()(const sys_time<_Duration>& __t) const
      -> decltype(time_point<_DestClock, _Duration>::clock::from_sys(__t)) {
    static_assert(__is_time_point_of_clock_v<decltype(_DestClock::from_sys(__t)), _DestClock>,
                  "DestClock::from_sys(t) must return a time_point of DestClock");
    return _DestClock::from_sys(__t);
  }
};

#  if _LIBCPP_HAS_EXPERIMENTAL_TZDB && _LIBCPP_HAS_TIME_ZONE_DATABASE && _LIBCPP_HAS_FILESYSTEM &&                     \
      _LIBCPP_HAS_LOCALIZATION

// [time.clock.cast.id]
template <>
struct clock_time_conversion<utc_clock, utc_clock> {
  template <class _Duration>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI utc_time<_Duration> operator()(const utc_time<_Duration>& __t) const {
    return __t;
  }
};

// [time.clock.cast.sys.utc]
template <>
struct clock_time_conversion<utc_clock, system_clock> {
  template <class _Duration>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI utc_time<common_type_t<_Duration, seconds>>
  operator()(const sys_time<_Duration>& __t) const {
    return utc_clock::from_sys(__t);
  }
};

template <>
struct clock_time_conversion<system_clock, utc_clock> {
  template <class _Duration>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI sys_time<common_type_t<_Duration, seconds>>
  operator()(const utc_time<_Duration>& __t) const {
    return utc_clock::to_sys(__t);
  }
};

// [time.clock.cast.utc]
template <class _SourceClock>
struct clock_time_conversion<utc_clock, _SourceClock> {
  template <class _Duration>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI auto operator()(const time_point<_SourceClock, _Duration>& __t) const
      -> decltype(time_point<_SourceClock, _Duration>::clock::to_utc(__t)) {
    static_assert(__is_time_point_of_clock_v<decltype(_SourceClock::to_utc(__t)), utc_clock>,
                  "SourceClock::to_utc(t) must return a utc_time");
    return _SourceClock::to_utc(__t);
  }
};

template <class _DestClock>
struct clock_time_conversion<_DestClock, utc_clock> {
  template <class _Duration>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI auto operator()(const utc_time<_Duration>& __t) const
      -> decltype(time_point<_DestClock, _Duration>::clock::from_utc(__t)) {
    static_assert(__is_time_point_of_clock_v<decltype(_DestClock::from_utc(__t)), _DestClock>,
                  "DestClock::from_utc(t) must return a time_point of DestClock");
    return _DestClock::from_utc(__t);
  }
};

#  endif // _LIBCPP_HAS_EXPERIMENTAL_TZDB && _LIBCPP_HAS_TIME_ZONE_DATABASE && _LIBCPP_HAS_FILESYSTEM &&
         // _LIBCPP_HAS_LOCALIZATION

// [time.clock.cast.fn]
//
// template<class DestClock, class SourceClock, class Duration>
//   auto clock_cast(const time_point<SourceClock, Duration>& t);
//
// Constraints: At least one of the following clock time conversion
// expressions is well-formed:
//   - clock_time_conversion<DestClock, SourceClock>{}(t)
//   - clock_time_conversion<DestClock, system_clock>{}(
//       clock_time_conversion<system_clock, SourceClock>{}(t))
//   - clock_time_conversion<DestClock, utc_clock>{}(
//       clock_time_conversion<utc_clock, SourceClock>{}(t))
//   - clock_time_conversion<DestClock, utc_clock>{}(
//       clock_time_conversion<utc_clock, system_clock>{}(
//         clock_time_conversion<system_clock, SourceClock>{}(t)))
//   - clock_time_conversion<DestClock, system_clock>{}(
//       clock_time_conversion<system_clock, utc_clock>{}(
//         clock_time_conversion<utc_clock, SourceClock>{}(t)))
// A clock time conversion expression is considered better than another clock
// time conversion expression if it involves fewer operator() calls on
// clock_time_conversion specializations.
//
// Mandates: Among the well-formed clock time conversion expressions from the
// above list, there is a unique best expression.
//
// Returns: The best well-formed clock time conversion expression in the above
// list.
template <class _DestClock, class _SourceClock, class _Duration>
concept __clock_cast_one_call = requires(const time_point<_SourceClock, _Duration>& __t) {
  clock_time_conversion<_DestClock, _SourceClock>{}(__t);
};

template <class _DestClock, class _MidClock, class _SourceClock, class _Duration>
concept __clock_cast_two_calls = requires(const time_point<_SourceClock, _Duration>& __t) {
  clock_time_conversion<_DestClock, _MidClock>{}(clock_time_conversion<_MidClock, _SourceClock>{}(__t));
};

template <class _DestClock, class _MidClock2, class _MidClock1, class _SourceClock, class _Duration>
concept __clock_cast_three_calls = requires(const time_point<_SourceClock, _Duration>& __t) {
  clock_time_conversion<_DestClock, _MidClock2>{}(
      clock_time_conversion<_MidClock2, _MidClock1>{}(clock_time_conversion<_MidClock1, _SourceClock>{}(__t)));
};

enum class __clock_cast_strategy {
  __none,
  __ambiguous_two_calls,
  __ambiguous_three_calls,
  __direct,
  __via_sys,
  __via_utc,
  __via_sys_then_utc,
  __via_utc_then_sys,
};

template <class _DestClock, class _SourceClock, class _Duration>
_LIBCPP_HIDE_FROM_ABI consteval __clock_cast_strategy __select_clock_cast_strategy() {
  if constexpr (__clock_cast_one_call<_DestClock, _SourceClock, _Duration>) {
    return __clock_cast_strategy::__direct;
  } else {
    constexpr bool __via_sys = __clock_cast_two_calls<_DestClock, system_clock, _SourceClock, _Duration>;
#  if _LIBCPP_HAS_EXPERIMENTAL_TZDB && _LIBCPP_HAS_TIME_ZONE_DATABASE && _LIBCPP_HAS_FILESYSTEM &&                     \
      _LIBCPP_HAS_LOCALIZATION
    constexpr bool __via_utc = __clock_cast_two_calls<_DestClock, utc_clock, _SourceClock, _Duration>;
    constexpr bool __via_sys_then_utc =
        __clock_cast_three_calls<_DestClock, utc_clock, system_clock, _SourceClock, _Duration>;
    constexpr bool __via_utc_then_sys =
        __clock_cast_three_calls<_DestClock, system_clock, utc_clock, _SourceClock, _Duration>;
#  else
    constexpr bool __via_utc          = false;
    constexpr bool __via_sys_then_utc = false;
    constexpr bool __via_utc_then_sys = false;
#  endif

    if constexpr (__via_sys && __via_utc)
      return __clock_cast_strategy::__ambiguous_two_calls;
    else if constexpr (__via_sys)
      return __clock_cast_strategy::__via_sys;
    else if constexpr (__via_utc)
      return __clock_cast_strategy::__via_utc;
    else if constexpr (__via_sys_then_utc && __via_utc_then_sys)
      return __clock_cast_strategy::__ambiguous_three_calls;
    else if constexpr (__via_sys_then_utc)
      return __clock_cast_strategy::__via_sys_then_utc;
    else if constexpr (__via_utc_then_sys)
      return __clock_cast_strategy::__via_utc_then_sys;
    else
      return __clock_cast_strategy::__none;
  }
}

template <class _DestClock, class _SourceClock, class _Duration>
  requires(chrono::__select_clock_cast_strategy<_DestClock, _SourceClock, _Duration>() != __clock_cast_strategy::__none)
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI auto clock_cast(const time_point<_SourceClock, _Duration>& __t) {
  constexpr __clock_cast_strategy __strategy =
      chrono::__select_clock_cast_strategy<_DestClock, _SourceClock, _Duration>();

  if constexpr (__strategy == __clock_cast_strategy::__ambiguous_two_calls)
    static_assert(false,
                  "clock_cast is ambiguous: SourceClock can be converted to DestClock both through system_clock and "
                  "through utc_clock, but the best conversion is required to be unique ([time.clock.cast.fn]/2). "
                  "Specialize clock_time_conversion<DestClock, SourceClock> to provide a direct conversion.");
  else if constexpr (__strategy == __clock_cast_strategy::__ambiguous_three_calls)
    static_assert(false,
                  "clock_cast is ambiguous: SourceClock can be converted to DestClock both through system_clock and "
                  "then utc_clock, and through utc_clock and then system_clock, but the best conversion is required "
                  "to be unique ([time.clock.cast.fn]/2). Specialize clock_time_conversion<DestClock, SourceClock> to "
                  "provide a direct conversion.");
  else if constexpr (__strategy == __clock_cast_strategy::__direct)
    return clock_time_conversion<_DestClock, _SourceClock>{}(__t);
  else if constexpr (__strategy == __clock_cast_strategy::__via_sys)
    return clock_time_conversion<_DestClock, system_clock>{}(clock_time_conversion<system_clock, _SourceClock>{}(__t));
#  if _LIBCPP_HAS_EXPERIMENTAL_TZDB && _LIBCPP_HAS_TIME_ZONE_DATABASE && _LIBCPP_HAS_FILESYSTEM &&                     \
      _LIBCPP_HAS_LOCALIZATION
  else if constexpr (__strategy == __clock_cast_strategy::__via_utc)
    return clock_time_conversion<_DestClock, utc_clock>{}(clock_time_conversion<utc_clock, _SourceClock>{}(__t));
  else if constexpr (__strategy == __clock_cast_strategy::__via_sys_then_utc)
    return clock_time_conversion<_DestClock, utc_clock>{}(
        clock_time_conversion<utc_clock, system_clock>{}(clock_time_conversion<system_clock, _SourceClock>{}(__t)));
  else if constexpr (__strategy == __clock_cast_strategy::__via_utc_then_sys)
    return clock_time_conversion<_DestClock, system_clock>{}(
        clock_time_conversion<system_clock, utc_clock>{}(clock_time_conversion<utc_clock, _SourceClock>{}(__t)));
#  endif
}

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___CHRONO_CLOCK_CAST_H
