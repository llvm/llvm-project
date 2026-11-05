//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_CLEAR_PADDING_H
#define _LIBCPP___ATOMIC_CLEAR_PADDING_H

#include <__config>
#include <__memory/addressof.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_cv.h>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_builtin(__builtin_clear_padding)

template <class _Tp>
inline const bool __needs_clear_padding_v =
    !__has_unique_object_representations(_Tp) && !is_same<_Tp, float>::value && !is_same<_Tp, double>::value;

template <class _Tp, __enable_if_t<!__needs_clear_padding_v<__remove_cv_t<_Tp> >, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR _Tp& __clear_padding_if_needed(_Tp& __obj) _NOEXCEPT {
  return __obj;
}

template <class _Tp, __enable_if_t<__needs_clear_padding_v<__remove_cv_t<_Tp> >, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR _Tp& __clear_padding_if_needed(_Tp& __obj) _NOEXCEPT {
  return __libcpp_is_constant_evaluated() ? __obj : (__builtin_clear_padding(std::addressof(__obj)), __obj);
}

// clang fails to inline the function when the memory order is a constant
template <class _Tp, class _Up, class _CasFunc, __enable_if_t<!__needs_clear_padding_v<__remove_cv_t<_Tp> >, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE bool
__atomic_cas_with_clear_padding(_Tp* __expected, _Up __value, _CasFunc&& __cas_func) {
  return __cas_func(__expected, __value);
}

template <class _Tp, class _Up, class _CasFunc, __enable_if_t<__needs_clear_padding_v<__remove_cv_t<_Tp> >, int> = 0>
_LIBCPP_HIDE_FROM_ABI bool __atomic_cas_with_clear_padding(_Tp* __expected, _Up __value, _CasFunc&& __cas_func) {
  std::__clear_padding_if_needed(__value);
  __remove_cv_t<_Tp> __expected_copy = *__expected;
  std::__clear_padding_if_needed(__expected_copy);
  if (__cas_func(std::addressof(__expected_copy), __value)) {
    return true;
  } else {
    std::memcpy(__expected, std::addressof(__expected_copy), sizeof(__remove_cv_t<_Tp>));
    return false;
  }
}

#else // __has_builtin(__builtin_clear_padding)

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR _Tp& __clear_padding_if_needed(_Tp& __obj) _NOEXCEPT {
  return __obj;
}

template <class _Tp, class _Up, class _CasFunc>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE bool
__atomic_cas_with_clear_padding(_Tp* __expected, _Up __value, _CasFunc&& __cas_func) {
  return __cas_func(__expected, __value);
}

#endif // __has_builtin(__builtin_clear_padding)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_CLEAR_PADDING_H
