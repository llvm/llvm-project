//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MATH_HYPOT_H
#define _LIBCPP___MATH_HYPOT_H

#include <__algorithm/max.h>
#include <__config>
#include <__math/abs.h>
#include <__math/roots.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_same.h>
#include <__type_traits/promote.h>
#include <array>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __math {

inline _LIBCPP_HIDE_FROM_ABI float hypot(float __x, float __y) _NOEXCEPT { return __builtin_hypotf(__x, __y); }

template <class = int>
_LIBCPP_HIDE_FROM_ABI double hypot(double __x, double __y) _NOEXCEPT {
  return __builtin_hypot(__x, __y);
}

inline _LIBCPP_HIDE_FROM_ABI long double hypot(long double __x, long double __y) _NOEXCEPT {
  return __builtin_hypotl(__x, __y);
}

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI typename __promote<_A1, _A2>::type hypot(_A1 __x, _A2 __y) _NOEXCEPT {
  using __result_type = typename __promote<_A1, _A2>::type;
  static_assert(!(_IsSame<_A1, __result_type>::value && _IsSame<_A2, __result_type>::value), "");
  return __math::hypot((__result_type)__x, (__result_type)__y);
}

#if _LIBCPP_STD_VER >= 17
// Factors needed to determine if over-/underflow might happen for `std::hypot(x,y,z)`.
template <class _Real>
struct __hypot_factors {
  _Real __threshold;
  _Real __scale_xyz;
  _Real __scale_M;
};

// Computes `__hypot_factors` needed to determine if over-/underflow might happen for `std::hypot(x,y,z)`.
// Returns: [underflow_factors, overflow_factors]
template <class _Real>
_LIBCPP_HIDE_FROM_ABI std::array<__math::__hypot_factors<_Real>, 2> __create_factors() {
  static_assert(std::numeric_limits<_Real>::is_iec559);

  __math::__hypot_factors<_Real> __underflow, __overflow;
  if constexpr (std::is_same_v<_Real, float>) {
    static_assert(-125 == std::numeric_limits<_Real>::min_exponent);
    static_assert(+128 == std::numeric_limits<_Real>::max_exponent);
    __underflow = __math::__hypot_factors<_Real>{0x1.0p-62f, 0x1.0p70f, 0x1.0p-70f};
    __overflow  = __math::__hypot_factors<_Real>{0x1.0p62f, 0x1.0p-70f, 0x1.0p+70f};
  } else if constexpr (std::is_same_v<_Real, double>) {
    static_assert(-1021 == std::numeric_limits<_Real>::min_exponent);
    static_assert(+1024 == std::numeric_limits<_Real>::max_exponent);
    __underflow = __math::__hypot_factors<_Real>{0x1.0p-510, 0x1.0p600, 0x1.0p-600};
    __overflow  = __math::__hypot_factors<_Real>{0x1.0p510, 0x1.0p-600, 0x1.0p+600};
  } else { // long double
    static_assert(std::is_same_v<_Real, long double>);
    if constexpr (sizeof(_Real) == sizeof(double))
      return static_cast<std::array<__math::__hypot_factors<_Real>, 2>>(__math::__create_factors<double>());
    else {
      static_assert(-16'381 == std::numeric_limits<_Real>::min_exponent);
      static_assert(+16'384 == std::numeric_limits<_Real>::max_exponent);
      __underflow = __math::__hypot_factors<_Real>{0x1.0p-8'190l, 0x1.0p9'000l, 0x1.0p-9'000l};
      __overflow  = __math::__hypot_factors<_Real>{0x1.0p8'190l, 0x1.0p-9'000l, 0x1.0p+9'000l};
    }
  }
  return {__underflow, __overflow};
}

// Computes the three-dimensional hypotenuse: `std::hypot(x,y,z)`.
// The naive implementation might over-/underflow which is why this implementation is more involved:
//    If the square of an argument might run into issues, we scale the arguments appropriately.
// See https://github.com/llvm/llvm-project/issues/92782 for a detailed discussion and summary.
template <class _Real>
_LIBCPP_HIDE_FROM_ABI _Real __hypot(_Real __x, _Real __y, _Real __z) {
  const auto [__underflow, __overflow] = __math::__create_factors<_Real>();
  _Real __M                            = std::max({__math::fabs(__x), __math::fabs(__y), __math::fabs(__z)});
  if (__M > __overflow.__threshold) { // x*x + y*y + z*z might overflow
    __x *= __overflow.__scale_xyz;
    __y *= __overflow.__scale_xyz;
    __z *= __overflow.__scale_xyz;
    __M = __overflow.__scale_M;
  } else if (__M < __underflow.__threshold) { // x*x + y*y + z*z might underflow
    __x *= __underflow.__scale_xyz;
    __y *= __underflow.__scale_xyz;
    __z *= __underflow.__scale_xyz;
    __M = __underflow.__scale_M;
  } else
    __M = 1;
  return __M * __math::sqrt(__x * __x + __y * __y + __z * __z);
}

inline _LIBCPP_HIDE_FROM_ABI float hypot(float __x, float __y, float __z) { return __math::__hypot(__x, __y, __z); }

inline _LIBCPP_HIDE_FROM_ABI double hypot(double __x, double __y, double __z) { return __math::__hypot(__x, __y, __z); }

inline _LIBCPP_HIDE_FROM_ABI long double hypot(long double __x, long double __y, long double __z) {
  return __math::__hypot(__x, __y, __z);
}

template <class _A1,
          class _A2,
          class _A3,
          std::enable_if_t< is_arithmetic_v<_A1> && is_arithmetic_v<_A2> && is_arithmetic_v<_A3>, int> = 0 >
_LIBCPP_HIDE_FROM_ABI typename __promote<_A1, _A2, _A3>::type hypot(_A1 __x, _A2 __y, _A3 __z) _NOEXCEPT {
  using __result_type = typename __promote<_A1, _A2, _A3>::type;
  static_assert(!(
      std::is_same_v<_A1, __result_type> && std::is_same_v<_A2, __result_type> && std::is_same_v<_A3, __result_type>));
  return __math::__hypot(static_cast<__result_type>(__x), static_cast<__result_type>(__y), static_cast<__result_type>(__z));
}
#endif

} // namespace __math

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MATH_HYPOT_H
