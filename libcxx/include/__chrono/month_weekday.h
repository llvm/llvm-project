// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_MONTH_WEEKDAY_H
#define _LIBCPP___CHRONO_MONTH_WEEKDAY_H

#include <__chrono/month.h>
#include <__chrono/weekday.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/hash.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

class month_weekday {
private:
  chrono::month __m_;
  chrono::weekday_indexed __wdi_;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr month_weekday(const chrono::month& __mval,
                                                const chrono::weekday_indexed& __wdival) noexcept
      : __m_{__mval}, __wdi_{__wdival} {}
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month month() const noexcept { return __m_; }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::weekday_indexed weekday_indexed() const noexcept {
    return __wdi_;
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept { return __m_.ok() && __wdi_.ok(); }
};

_LIBCPP_HIDE_FROM_ABI inline constexpr bool
operator==(const month_weekday& __lhs, const month_weekday& __rhs) noexcept {
  return __lhs.month() == __rhs.month() && __lhs.weekday_indexed() == __rhs.weekday_indexed();
}

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr month_weekday
operator/(const month& __lhs, const weekday_indexed& __rhs) noexcept {
  return month_weekday{__lhs, __rhs};
}

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr month_weekday
operator/(int __lhs, const weekday_indexed& __rhs) noexcept {
  return month_weekday{month(__lhs), __rhs};
}

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr month_weekday
operator/(const weekday_indexed& __lhs, const month& __rhs) noexcept {
  return month_weekday{__rhs, __lhs};
}

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr month_weekday
operator/(const weekday_indexed& __lhs, int __rhs) noexcept {
  return month_weekday{month(__rhs), __lhs};
}

class month_weekday_last {
  chrono::month __m_;
  chrono::weekday_last __wdl_;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr month_weekday_last(const chrono::month& __mval,
                                                     const chrono::weekday_last& __wdlval) noexcept
      : __m_{__mval}, __wdl_{__wdlval} {}
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month month() const noexcept { return __m_; }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::weekday_last weekday_last() const noexcept {
    return __wdl_;
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept { return __m_.ok() && __wdl_.ok(); }
};

_LIBCPP_HIDE_FROM_ABI inline constexpr bool
operator==(const month_weekday_last& __lhs, const month_weekday_last& __rhs) noexcept {
  return __lhs.month() == __rhs.month() && __lhs.weekday_last() == __rhs.weekday_last();
}

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr month_weekday_last
operator/(const month& __lhs, const weekday_last& __rhs) noexcept {
  return month_weekday_last{__lhs, __rhs};
}

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr month_weekday_last
operator/(int __lhs, const weekday_last& __rhs) noexcept {
  return month_weekday_last{month(__lhs), __rhs};
}

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr month_weekday_last
operator/(const weekday_last& __lhs, const month& __rhs) noexcept {
  return month_weekday_last{__rhs, __lhs};
}

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr month_weekday_last
operator/(const weekday_last& __lhs, int __rhs) noexcept {
  return month_weekday_last{month(__rhs), __lhs};
}
} // namespace chrono

#  if _LIBCPP_STD_VER >= 26

template <>
struct hash<chrono::month_weekday> {
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static size_t operator()(const chrono::month_weekday& __mw) noexcept {
    return std::__hash_combine(
        hash<chrono::month>{}(__mw.month()), hash<chrono::weekday_indexed>{}(__mw.weekday_indexed()));
  }
};

template <>
struct hash<chrono::month_weekday_last> {
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static size_t operator()(const chrono::month_weekday_last& __mwl) noexcept {
    return std::__hash_combine(
        hash<chrono::month>{}(__mwl.month()), hash<chrono::weekday_last>{}(__mwl.weekday_last()));
  }
};

#  endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___CHRONO_MONTH_WEEKDAY_H
