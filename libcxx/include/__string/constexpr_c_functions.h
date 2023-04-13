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
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

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

template <class _Tp, class _Up>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 int
__constexpr_memcmp(const _Tp* __lhs, const _Up* __rhs, size_t __count) {
  static_assert(__libcpp_is_trivially_equality_comparable<_Tp, _Up>::value,
                "_Tp and _Up have to be trivially equality comparable");

  if (__libcpp_is_constant_evaluated()) {
#ifdef _LIBCPP_COMPILER_CLANG_BASED
    if (sizeof(_Tp) == 1 && !is_same<_Tp, bool>::value)
      return __builtin_memcmp(__lhs, __rhs, __count);
#endif

    while (__count != 0) {
      if (*__lhs < *__rhs)
        return -1;
      if (*__rhs < *__lhs)
        return 1;

      __count -= sizeof(_Tp);
      ++__lhs;
      ++__rhs;
    }
    return 0;
  } else {
    return __builtin_memcmp(__lhs, __rhs, __count);
  }
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 const char*
__constexpr_char_memchr(const char* __str, int __char, size_t __count) {
#if __has_builtin(__builtin_char_memchr)
  return __builtin_char_memchr(__str, __char, __count);
#else
  if (!__libcpp_is_constant_evaluated())
    return static_cast<const char*>(__builtin_memchr(__str, __char, __count));
  for (; __count; --__count) {
    if (*__str == __char)
      return __str;
    ++__str;
  }
  return nullptr;
#endif
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___STRING_CONSTEXPR_C_FUNCTIONS_H
