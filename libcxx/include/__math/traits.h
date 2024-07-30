//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MATH_TRAITS_H
#define _LIBCPP___MATH_TRAITS_H

#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_floating_point.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_signed.h>
#include <__type_traits/promote.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __math {

// signbit

template <class _A1, __enable_if_t<is_floating_point<_A1>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool signbit(_A1 __x) _NOEXCEPT {
  return __builtin_signbit(__x);
}

template <class _A1, __enable_if_t<is_integral<_A1>::value && is_signed<_A1>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool signbit(_A1 __x) _NOEXCEPT {
  return __x < 0;
}

template <class _A1, __enable_if_t<is_integral<_A1>::value && !is_signed<_A1>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool signbit(_A1) _NOEXCEPT {
  return false;
}

// isfinite

template <class _A1, __enable_if_t<is_arithmetic<_A1>::value && numeric_limits<_A1>::has_infinity, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isfinite(_A1 __x) _NOEXCEPT {
  return __builtin_isfinite((typename __promote<_A1>::type)__x);
}

template <class _A1, __enable_if_t<is_arithmetic<_A1>::value && !numeric_limits<_A1>::has_infinity, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isfinite(_A1) _NOEXCEPT {
  return true;
}

_LIBCPP_NODISCARD inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isfinite(float __x) _NOEXCEPT {
  return __builtin_isfinite(__x);
}

_LIBCPP_NODISCARD inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isfinite(double __x) _NOEXCEPT {
  return __builtin_isfinite(__x);
}

_LIBCPP_NODISCARD inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isfinite(long double __x) _NOEXCEPT {
  return __builtin_isfinite(__x);
}

// isinf

template <class _A1, __enable_if_t<is_floating_point<_A1>::value, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isinf(_A1 __x) _NOEXCEPT {
  return __builtin_isinf((typename __promote<_A1>::type)__x);
}

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isinf(_A1) _NOEXCEPT {
  return false;
}

_LIBCPP_NODISCARD inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isinf(float __x) _NOEXCEPT {
  return __builtin_isinf(__x);
}

_LIBCPP_NODISCARD inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI
// When libc++ is layered on top of glibc on Linux, glibc's `math.h` is included. When compiling with
// `-std=c++03`, this header brings the function declaration of `isinf(double)` with return type of
// `int` into scope. This differs from the C99 standard as only a macro of the form `#define isinf(arg)`
// is expected. Therefore, libc++ needs to respect the presense of this `double` overload with return type
// `int` and cannot redefine it with return type `bool` like it is supposed to be, as it will conflict with
// the declaration already in scope. For `-std=c++11` and beyond this issue is fixed, as glibc guards the
// `double` overload of `isinf` by preprocessor macros.
//
// When libc++ is layered on top of Bionic's libc, `math.h` exposes a function prototype for `isnan(double)`
// with return type `int`. This function prototype in Bionic's libc is not guarded by any preprocessor macros
// and will therefore conflict.
//
// `_LIBCPP_PREFERRED_OVERLOAD` specifies that a given overload is a better match than an otherwise equally good
// function declaration. This is implemented in modern versions of Clang via `__attribute__((__enable_if__))`, and
// not elsewhere. See https://github.com/llvm/llvm-project/commit/5fd17ab1b093f6b59aabb27f6c2c2278e65c2707 for
// details. We use `_LIBCPP_PREFERRED_OVERLOAD` to define overloads in the global namespace that displace the
// overloads provided by the C libraries mentioned above.
#if ((defined(_LIBCPP_CXX03_LANG) && defined(__GLIBC__)) || defined(__BIONIC__)) && defined(_LIBCPP_PREFERRED_OVERLOAD)
_LIBCPP_PREFERRED_OVERLOAD
#endif
    bool
    isinf(double __x) _NOEXCEPT {
  return __builtin_isinf(__x);
}

_LIBCPP_NODISCARD inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isinf(long double __x) _NOEXCEPT {
  return __builtin_isinf(__x);
}

// isnan

template <class _A1, __enable_if_t<is_floating_point<_A1>::value, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isnan(_A1 __x) _NOEXCEPT {
  return __builtin_isnan(__x);
}

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isnan(_A1) _NOEXCEPT {
  return false;
}

_LIBCPP_NODISCARD inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isnan(float __x) _NOEXCEPT {
  return __builtin_isnan(__x);
}

_LIBCPP_NODISCARD inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI
// When libc++ is layered on top of glibc on Linux, glibc's `math.h` is included. When compiling with
// `-std=c++03`, this header brings the function declaration of `isnan(double)` with return type of
// `int` into scope. This differs from the C99 standard as only a macro of the form `#define isnan(arg)`
// is expected. Therefore, libc++ needs to respect the presense of this `double` overload with return type
// `int` and cannot redefine it with return type `bool` like it is supposed to be, as it will conflict with
// the declaration already in scope. For `-std=c++11` and beyond this issue is fixed, as glibc guards the
// `double` overload of `isnan` by preprocessor macros.
//
// When libc++ is layered on top of Bionic's libc, `math.h` exposes a function prototype for `isnan(double)`
// with return type `int`. This function prototype in Bionic's libc is not guarded by any preprocessor macros
// and will therefore conflict.
//
// `_LIBCPP_PREFERRED_OVERLOAD` specifies that a given overload is a better match than an otherwise equally good
// function declaration. This is implemented in modern versions of Clang via `__attribute__((__enable_if__))`, and
// not elsewhere. See https://github.com/llvm/llvm-project/commit/5fd17ab1b093f6b59aabb27f6c2c2278e65c2707 for
// details. We use `_LIBCPP_PREFERRED_OVERLOAD` to define overloads in the global namespace that displace the
// overloads provided by the C libraries mentioned above.
#if ((defined(_LIBCPP_CXX03_LANG) && defined(__GLIBC__)) || defined(__BIONIC__)) && defined(_LIBCPP_PREFERRED_OVERLOAD)
_LIBCPP_PREFERRED_OVERLOAD
#endif
    bool
    isnan(double __x) _NOEXCEPT {
  return __builtin_isnan(__x);
}

_LIBCPP_NODISCARD inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isnan(long double __x) _NOEXCEPT {
  return __builtin_isnan(__x);
}

// isnormal

template <class _A1, __enable_if_t<is_floating_point<_A1>::value, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isnormal(_A1 __x) _NOEXCEPT {
  return __builtin_isnormal(__x);
}

template <class _A1, __enable_if_t<is_integral<_A1>::value, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI bool isnormal(_A1 __x) _NOEXCEPT {
  return __x != 0;
}

// isgreater

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool isgreater(_A1 __x, _A2 __y) _NOEXCEPT {
  using type = typename __promote<_A1, _A2>::type;
  return __builtin_isgreater((type)__x, (type)__y);
}

// isgreaterequal

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool isgreaterequal(_A1 __x, _A2 __y) _NOEXCEPT {
  using type = typename __promote<_A1, _A2>::type;
  return __builtin_isgreaterequal((type)__x, (type)__y);
}

// isless

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool isless(_A1 __x, _A2 __y) _NOEXCEPT {
  using type = typename __promote<_A1, _A2>::type;
  return __builtin_isless((type)__x, (type)__y);
}

// islessequal

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool islessequal(_A1 __x, _A2 __y) _NOEXCEPT {
  using type = typename __promote<_A1, _A2>::type;
  return __builtin_islessequal((type)__x, (type)__y);
}

// islessgreater

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool islessgreater(_A1 __x, _A2 __y) _NOEXCEPT {
  using type = typename __promote<_A1, _A2>::type;
  return __builtin_islessgreater((type)__x, (type)__y);
}

// isunordered

template <class _A1, class _A2, __enable_if_t<is_arithmetic<_A1>::value && is_arithmetic<_A2>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI bool isunordered(_A1 __x, _A2 __y) _NOEXCEPT {
  using type = typename __promote<_A1, _A2>::type;
  return __builtin_isunordered((type)__x, (type)__y);
}

} // namespace __math

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MATH_TRAITS_H
