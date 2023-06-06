//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___STRING_CONSTEXPR_C_FUNCTIONS_H
#define _LIBCPP___STRING_CONSTEXPR_C_FUNCTIONS_H

#include <__config>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_equality_comparable.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_lexicographically_comparable.h>
#include <__type_traits/remove_cv.h>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// Type used to encode that a function takes an integer that represents a number
// of elements as opposed to a number of bytes.
enum class __element_count : size_t {};

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 size_t __constexpr_strlen(const char* __str) {
  // GCC currently doesn't support __builtin_strlen for heap-allocated memory during constant evaluation.
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70816
#ifdef _LIBCPP_COMPILER_GCC
  if (__libcpp_is_constant_evaluated()) {
    size_t __i = 0;
    for (; __str[__i] != '\0'; ++__i)
      ;
    return __i;
  }
#endif
  return __builtin_strlen(__str);
}

// Because of __libcpp_is_trivially_lexicographically_comparable we know that comparing the object representations is
// equivalent to a std::memcmp. Since we have multiple objects contiguously in memory, we can call memcmp once instead
// of invoking it on every object individually.
template <class _Tp, class _Up>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 int
__constexpr_memcmp(const _Tp* __lhs, const _Up* __rhs, __element_count __n) {
  static_assert(__libcpp_is_trivially_lexicographically_comparable<_Tp, _Up>::value,
                "_Tp and _Up have to be trivially lexicographically comparable");

  auto __count = static_cast<size_t>(__n);

  if (__libcpp_is_constant_evaluated()) {
#ifdef _LIBCPP_COMPILER_CLANG_BASED
    if (sizeof(_Tp) == 1 && !is_same<_Tp, bool>::value)
      return __builtin_memcmp(__lhs, __rhs, __count * sizeof(_Tp));
#endif

    while (__count != 0) {
      if (*__lhs < *__rhs)
        return -1;
      if (*__rhs < *__lhs)
        return 1;

      --__count;
      ++__lhs;
      ++__rhs;
    }
    return 0;
  } else {
    return __builtin_memcmp(__lhs, __rhs, __count * sizeof(_Tp));
  }
}

// Because of __libcpp_is_trivially_equality_comparable we know that comparing the object representations is equivalent
// to a std::memcmp(...) == 0. Since we have multiple objects contiguously in memory, we can call memcmp once instead
// of invoking it on every object individually.
template <class _Tp, class _Up>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 bool
__constexpr_memcmp_equal(const _Tp* __lhs, const _Up* __rhs, __element_count __n) {
  static_assert(__libcpp_is_trivially_equality_comparable<_Tp, _Up>::value,
                "_Tp and _Up have to be trivially equality comparable");

  auto __count = static_cast<size_t>(__n);

  if (__libcpp_is_constant_evaluated()) {
#ifdef _LIBCPP_COMPILER_CLANG_BASED
    if (sizeof(_Tp) == 1 && is_integral<_Tp>::value && !is_same<_Tp, bool>::value)
      return __builtin_memcmp(__lhs, __rhs, __count * sizeof(_Tp)) == 0;
#endif
    while (__count != 0) {
      if (*__lhs != *__rhs)
        return false;

      --__count;
      ++__lhs;
      ++__rhs;
    }
    return true;
  } else {
    return __builtin_memcmp(__lhs, __rhs, __count * sizeof(_Tp)) == 0;
  }
}

template <class _Tp, class _Up>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Tp* __constexpr_memchr(_Tp* __str, _Up __value, size_t __count) {
  static_assert(sizeof(_Tp) == 1 && __libcpp_is_trivially_equality_comparable<_Tp, _Up>::value,
                "Calling memchr on non-trivially equality comparable types is unsafe.");

  if (__libcpp_is_constant_evaluated()) {
// use __builtin_char_memchr to optimize constexpr evaluation if we can
#if _LIBCPP_STD_VER >= 17 && __has_builtin(__builtin_char_memchr)
    if constexpr (is_same_v<remove_cv_t<_Tp>, char> && is_same_v<remove_cv_t<_Up>, char>)
      return __builtin_char_memchr(__str, __value, __count);
#endif

    for (; __count; --__count) {
      if (*__str == __value)
        return __str;
      ++__str;
    }
    return nullptr;
  } else {
    char __value_buffer = 0;
    __builtin_memcpy(&__value_buffer, &__value, sizeof(char));
    return static_cast<_Tp*>(__builtin_memchr(__str, __value_buffer, __count));
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___STRING_CONSTEXPR_C_FUNCTIONS_H
