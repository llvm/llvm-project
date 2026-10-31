//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_CHECKED_INT_H
#define _LIBCPP___UTILITY_CHECKED_INT_H

#include <__config>
#include <__memory/addressof.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_signed.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _From, class _To>
inline const bool __is_lossless_conversion_v =
    is_signed<_From>::value == is_signed<_To>::value
        ? sizeof(_From) <= sizeof(_To)
        : is_signed<_To>::value && sizeof(_From) < sizeof(_To);

template <class _IntT>
class __checked_int {
  _IntT __value_     = 0;
  bool __overflowed_ = false;

  __checked_int() = default;

public:
  _LIBCPP_CONSTEXPR explicit __checked_int(_IntT __value) : __value_(__value) {}

  _LIBCPP_CONSTEXPR _IntT __value() const { return __value_; }
  _LIBCPP_CONSTEXPR bool __overflowed() const { return __overflowed_; }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int& operator+=(__checked_int& __lhs, __checked_int __rhs) {
    __lhs.__overflowed_ |=
        __rhs.__overflowed_ | __builtin_add_overflow(__lhs.__value_, __rhs.__value_, std::addressof(__lhs.__value_));
    return __lhs;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int& operator-=(__checked_int& __lhs, __checked_int __rhs) {
    __lhs.__overflowed_ |=
        __rhs.__overflowed_ | __builtin_sub_overflow(__lhs.__value_, __rhs.__value_, std::addressof(__lhs.__value_));
    return __lhs;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int& operator*=(__checked_int& __lhs, __checked_int __rhs) {
    __lhs.__overflowed_ |=
        __rhs.__overflowed_ | __builtin_mul_overflow(__lhs.__value_, __rhs.__value_, std::addressof(__lhs.__value_));
    return __lhs;
  }

  template <class _OtherInt, __enable_if_t<__is_lossless_conversion_v<_OtherInt, _IntT>, int> = 0>
  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int& operator+=(__checked_int& __lhs, _OtherInt __rhs) {
    return __lhs += __checked_int(__rhs);
  }

  template <class _OtherInt, __enable_if_t<__is_lossless_conversion_v<_OtherInt, _IntT>, int> = 0>
  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int& operator-=(__checked_int& __lhs, _OtherInt __rhs) {
    return __lhs -= __checked_int(__rhs);
  }

  template <class _OtherInt, __enable_if_t<__is_lossless_conversion_v<_OtherInt, _IntT>, int> = 0>
  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int& operator*=(__checked_int& __lhs, _OtherInt __rhs) {
    return __lhs *= __checked_int(__rhs);
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int operator+(__checked_int __lhs, __checked_int __rhs) {
    auto __copy = __lhs;
    __copy += __rhs;
    return __copy;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int operator-(__checked_int __lhs, __checked_int __rhs) {
    auto __copy = __lhs;
    __copy -= __rhs;
    return __copy;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int operator*(__checked_int __lhs, __checked_int __rhs) {
    auto __copy = __lhs;
    __copy *= __rhs;
    return __copy;
  }

  template <class _OtherInt, __enable_if_t<__is_lossless_conversion_v<_OtherInt, _IntT>, int> = 0>
  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int operator+(__checked_int __lhs, _OtherInt __rhs) {
    return __lhs + __checked_int(__rhs);
  }

  template <class _OtherInt, __enable_if_t<__is_lossless_conversion_v<_OtherInt, _IntT>, int> = 0>
  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int operator-(__checked_int __lhs, _OtherInt __rhs) {
    return __lhs - __checked_int(__rhs);
  }

  template <class _OtherInt, __enable_if_t<__is_lossless_conversion_v<_OtherInt, _IntT>, int> = 0>
  _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __checked_int operator*(__checked_int __lhs, _OtherInt __rhs) {
    return __lhs * __checked_int(__rhs);
  }
};

_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(__checked_int);

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_CHECKED_INT_H
