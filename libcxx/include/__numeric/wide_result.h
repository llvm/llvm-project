//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NUMERIC_WIDE_RESULT_H
#define _LIBCPP___NUMERIC_WIDE_RESULT_H

#include <__compare/ordering.h>
#include <__config>
#include <__type_traits/conditional.h>
#include <__type_traits/integer_traits.h>
#include <__type_traits/uint_fast_bits.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 29

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
class wide_result {
public:
  static_assert(__is_unsigned_integer_v<_Tp>);

#  ifdef _LIBCPP_TESTING_CLMUL_NO_WIDER_TYPE
  using _Wide = void;
#  else
  using _Wide = __uint_fast_bits_t<2 * numeric_limits<_Tp>::digits>;
#  endif

  constexpr wide_result(_Tp __high, _Tp __low) noexcept
      :
#  ifdef _LIBCPP_LITTLE_ENDIAN
        __low_(__low),
        __high_(__high) {
  }
#  else
        __high_(__high),
        __low_(__low) {
  }
#  endif

  [[nodiscard]] constexpr _Tp high() const noexcept { return __high_; }
  [[nodiscard]] constexpr _Tp low() const noexcept { return __low_; }

  friend bool operator==(const wide_result&, const wide_result&) = default;
  friend constexpr strong_ordering operator<=>(const wide_result& __x, const wide_result& __y) noexcept {
    if constexpr (!is_same_v<_Wide, void>) {
      // Lambda is generic to prevent eager instantiation of return statement.
      constexpr auto __to_integer = []<class _Result = _Wide>(_Tp __high, _Tp __low) -> _Result {
        constexpr int __width = numeric_limits<_Tp>::digits;
        return (static_cast<_Result>(__high) << __width) | static_cast<_Result>(__low);
      };
      return __to_integer(__x.__high_, __x.__low_) <=> __to_integer(__y.__high_, __y.__low_);
    } else {
      if (__x.__high_ < __y.__high_)
        return strong_ordering::less;
      if (__x.__high_ > __y.__high_)
        return strong_ordering::greater;
      return __x.__low_ <=> __y.__low_;
    }
  }

private:
#  ifdef _LIBCPP_LITTLE_ENDIAN
  _Tp __low_;
  _Tp __high_;
#  else
  _Tp __high_;
  _Tp __low_;
#  endif
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 29

#endif // _LIBCPP___NUMERIC_WIDE_RESULT_H
