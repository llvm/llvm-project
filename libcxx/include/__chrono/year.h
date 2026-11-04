// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_YEAR_H
#define _LIBCPP___CHRONO_YEAR_H

#include <__chrono/duration.h>
#include <__compare/ordering.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/hash.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

class year {
private:
  short __y_;

public:
  year() = default;
  explicit inline constexpr year(int __val) noexcept : __y_(static_cast<short>(__val)) {}

  inline constexpr year& operator++() noexcept {
    ++__y_;
    return *this;
  }
  inline constexpr year operator++(int) noexcept {
    year __tmp = *this;
    ++(*this);
    return __tmp;
  }
  inline constexpr year& operator--() noexcept {
    --__y_;
    return *this;
  }
  inline constexpr year operator--(int) noexcept {
    year __tmp = *this;
    --(*this);
    return __tmp;
  }
  constexpr year& operator+=(const years& __dy) noexcept;
  constexpr year& operator-=(const years& __dy) noexcept;
  [[nodiscard]] inline constexpr year operator+() const noexcept { return *this; }
  [[nodiscard]] inline constexpr year operator-() const noexcept { return year{-__y_}; }

  [[nodiscard]] inline constexpr bool is_leap() const noexcept {
    return __y_ % 4 == 0 && (__y_ % 100 != 0 || __y_ % 400 == 0);
  }
  explicit inline constexpr operator int() const noexcept { return __y_; }
  [[nodiscard]] constexpr bool ok() const noexcept;
  [[nodiscard]] static inline constexpr year min() noexcept { return year{-32767}; }
  [[nodiscard]] static inline constexpr year max() noexcept { return year{32767}; }
};

inline constexpr bool operator==(const year& __lhs, const year& __rhs) noexcept {
  return static_cast<int>(__lhs) == static_cast<int>(__rhs);
}

constexpr strong_ordering operator<=>(const year& __lhs, const year& __rhs) noexcept {
  return static_cast<int>(__lhs) <=> static_cast<int>(__rhs);
}

[[nodiscard]] inline constexpr year operator+(const year& __lhs, const years& __rhs) noexcept {
  return year(static_cast<int>(__lhs) + __rhs.count());
}

[[nodiscard]] inline constexpr year operator+(const years& __lhs, const year& __rhs) noexcept { return __rhs + __lhs; }

[[nodiscard]] inline constexpr year operator-(const year& __lhs, const years& __rhs) noexcept { return __lhs + -__rhs; }

[[nodiscard]] inline constexpr years operator-(const year& __lhs, const year& __rhs) noexcept {
  return years{static_cast<int>(__lhs) - static_cast<int>(__rhs)};
}

inline constexpr year& year::operator+=(const years& __dy) noexcept {
  *this = *this + __dy;
  return *this;
}

inline constexpr year& year::operator-=(const years& __dy) noexcept {
  *this = *this - __dy;
  return *this;
}

constexpr bool year::ok() const noexcept {
  static_assert(static_cast<int>(std::numeric_limits<decltype(__y_)>::max()) == static_cast<int>(max()));
  return static_cast<int>(min()) <= __y_;
}

} // namespace chrono

#  if _LIBCPP_STD_VER >= 26

template <>
struct hash<chrono::year> {
  [[nodiscard]] static size_t operator()(const chrono::year& __y) noexcept { return static_cast<int>(__y); }
};

#  endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CHRONO_YEAR_H
