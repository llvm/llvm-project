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
#include <__concepts/same_as.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17

// Convert a chrono calendar time point to the given tm type,
// which must have the same properties as std::tm.
template <class _Tm, class _ChronoCalendarTimePoint>
_LIBCPP_HIDE_FROM_ABI _Tm __convert_to_tm(const _ChronoCalendarTimePoint& __value) {
  _Tm __result = {};
#  ifdef __GLIBC__
  __result.tm_zone = "UTC";
#  endif

  if constexpr (same_as<_ChronoCalendarTimePoint, chrono::day>)
    __result.tm_mday = static_cast<unsigned>(__value);
  else
    static_assert(sizeof(_ChronoCalendarTimePoint) == 0, "Add the missing type specialization");

  return __result;
}

#endif //if _LIBCPP_STD_VER > 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CHRONO_CONVERT_TO_TM_H
