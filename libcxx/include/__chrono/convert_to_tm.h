// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_CONVERT_TO_TM_H
#define _LIBCPP___CHRONO_CONVERT_TO_TM_H

#include <__chrono/day.h>
#include <__chrono/duration.h>
#include <__chrono/month.h>
#include <__chrono/weekday.h>
#include <__chrono/year.h>
#include <__concepts/same_as.h>
#include <__config>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17

// Convert a chrono (calendar) time point, or dururation to the given _Tm type,
// which must have the same properties as std::tm.
template <class _Tm, class _ChronoT>
_LIBCPP_HIDE_FROM_ABI _Tm __convert_to_tm(const _ChronoT& __value) {
  _Tm __result = {};
#  ifdef __GLIBC__
  __result.tm_zone = "UTC";
#  endif

  if constexpr (chrono::__is_duration<_ChronoT>::value) {
    // [time.format]/6
    //   ...  However, if a flag refers to a "time of day" (e.g. %H, %I, %p,
    //   etc.), then a specialization of duration is interpreted as the time of
    //   day elapsed since midnight.
    uint64_t __sec = chrono::duration_cast<chrono::seconds>(__value).count();
    __sec %= 24 * 3600;
    __result.tm_hour = __sec / 3600;
    __sec %= 3600;
    __result.tm_min = __sec / 60;
    __result.tm_sec = __sec % 60;
  } else if constexpr (same_as<_ChronoT, chrono::day>)
    __result.tm_mday = static_cast<unsigned>(__value);
  else if constexpr (same_as<_ChronoT, chrono::month>)
    __result.tm_mon = static_cast<unsigned>(__value) - 1;
  else if constexpr (same_as<_ChronoT, chrono::year>)
    __result.tm_year = static_cast<int>(__value) - 1900;
  else if constexpr (same_as<_ChronoT, chrono::weekday>)
    __result.tm_wday = __value.c_encoding();
  else
    static_assert(sizeof(_ChronoT) == 0, "Add the missing type specialization");

  return __result;
}

#endif //if _LIBCPP_STD_VER > 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CHRONO_CONVERT_TO_TM_H
