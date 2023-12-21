//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_SUPPORT_COMMON_H
#define _LIBCPP___ATOMIC_SUPPORT_COMMON_H

#include <__config>
#include <__memory/addressof.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/has_unique_object_representation.h>
#include <__type_traits/is_same.h>
#include <__type_traits/negation.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20 && __has_builtin(__builtin_clear_padding)

template <class _Tp>
struct __needs_clear_padding
    : _And<_Not<has_unique_object_representations<_Tp>>, _Not<is_same<_Tp, float>>, _Not<is_same<_Tp, double>>> {};

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr void __clear_padding_if_needed(_Tp&& __obj) noexcept {
  if constexpr (__needs_clear_padding<remove_cvref_t<_Tp>>::value) {
    if (!__builtin_is_constant_evaluated()) {
      __builtin_clear_padding(std::addressof(__obj));
    }
  }
}

template <class _Tp, class _Up, class _CasFunc>
_LIBCPP_HIDE_FROM_ABI bool __atomic_cas_with_clear_padding(_Tp* __expected, _Up&& __value, _CasFunc&& __cas_func) {
  if constexpr (!__needs_clear_padding<remove_cv_t<_Tp>>::value) {
    return __cas_func(__expected, std::forward<_Up>(__value));
  } else {
    std::__clear_padding_if_needed(__value);
    remove_cv_t<_Tp> __expected_copy = *__expected;
    std::__clear_padding_if_needed(__expected_copy);
    if (__cas_func(std::addressof(__expected_copy), std::forward<_Up>(__value))) {
      return true;
    } else {
      std::memcpy(__expected, std::addressof(__expected_copy), sizeof(remove_cv_t<_Tp>));
      return false;
    }
  }
}

#else // _LIBCPP_STD_VER >= 20 && __has_builtin(__builtin_clear_padding)

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR void __clear_padding_if_needed(_Tp&&) noexcept {}

template <class _Tp, class _Up, class _CasFunc>
_LIBCPP_HIDE_FROM_ABI bool __atomic_cas_with_clear_padding(_Tp* __expected, _Up&& __value, _CasFunc&& __cas_func) {
  return __cas_func(__expected, std::forward<_Up>(__value));
}

#endif // _LIBCPP_STD_VER >= 20 && __has_builtin(__builtin_clear_padding)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_SUPPORT_COMMON_H
