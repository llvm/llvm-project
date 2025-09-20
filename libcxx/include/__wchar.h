// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Lack of header guards is intentional.
// This file should only ever be included by wchar.h, which provides its own
// header guard. This prevents macro hiding issues with modules where cwchar
// complains that _LIBCPP_WCHAR_H isn't present.

#include <__config>
#include <__mbstate_t.h> // provide mbstate_t
#include <stddef.h>      // provide size_t

// include_next doesn't work with modules.
#if __has_include_next(<wchar.h>)
#  define _LIBCPP_INCLUDE_NEXT_WCHAR
#  include_next <wchar.h>
#  undef _LIBCPP_INCLUDE_NEXT_WCHAR
#endif

// Determine whether we have const-correct overloads for wcschr and friends.
#if defined(_WCHAR_H_CPLUSPLUS_98_CONFORMANCE_)
#  define _LIBCPP_WCHAR_H_HAS_CONST_OVERLOADS 1
#elif defined(__GLIBC_PREREQ)
#  if __GLIBC_PREREQ(2, 10)
#    define _LIBCPP_WCHAR_H_HAS_CONST_OVERLOADS 1
#  endif
#elif defined(_LIBCPP_MSVCRT)
#  if defined(_CRT_CONST_CORRECT_OVERLOADS)
#    define _LIBCPP_WCHAR_H_HAS_CONST_OVERLOADS 1
#  endif
#endif

#if _LIBCPP_HAS_WIDE_CHARACTERS
#  if defined(__cplusplus) && !defined(_LIBCPP_WCHAR_H_HAS_CONST_OVERLOADS) && defined(_LIBCPP_PREFERRED_OVERLOAD)
extern "C++" {
inline _LIBCPP_HIDE_FROM_ABI wchar_t* __libcpp_wcschr(const wchar_t* __s, wchar_t __c) {
  return (wchar_t*)wcschr(__s, __c);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD const wchar_t* wcschr(const wchar_t* __s, wchar_t __c) {
  return __libcpp_wcschr(__s, __c);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD wchar_t* wcschr(wchar_t* __s, wchar_t __c) {
  return __libcpp_wcschr(__s, __c);
}

inline _LIBCPP_HIDE_FROM_ABI wchar_t* __libcpp_wcspbrk(const wchar_t* __s1, const wchar_t* __s2) {
  return (wchar_t*)wcspbrk(__s1, __s2);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD const wchar_t*
wcspbrk(const wchar_t* __s1, const wchar_t* __s2) {
  return __libcpp_wcspbrk(__s1, __s2);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD wchar_t* wcspbrk(wchar_t* __s1, const wchar_t* __s2) {
  return __libcpp_wcspbrk(__s1, __s2);
}

inline _LIBCPP_HIDE_FROM_ABI wchar_t* __libcpp_wcsrchr(const wchar_t* __s, wchar_t __c) {
  return (wchar_t*)wcsrchr(__s, __c);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD const wchar_t* wcsrchr(const wchar_t* __s, wchar_t __c) {
  return __libcpp_wcsrchr(__s, __c);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD wchar_t* wcsrchr(wchar_t* __s, wchar_t __c) {
  return __libcpp_wcsrchr(__s, __c);
}

inline _LIBCPP_HIDE_FROM_ABI wchar_t* __libcpp_wcsstr(const wchar_t* __s1, const wchar_t* __s2) {
  return (wchar_t*)wcsstr(__s1, __s2);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD const wchar_t*
wcsstr(const wchar_t* __s1, const wchar_t* __s2) {
  return __libcpp_wcsstr(__s1, __s2);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD wchar_t* wcsstr(wchar_t* __s1, const wchar_t* __s2) {
  return __libcpp_wcsstr(__s1, __s2);
}

inline _LIBCPP_HIDE_FROM_ABI wchar_t* __libcpp_wmemchr(const wchar_t* __s, wchar_t __c, size_t __n) {
  return (wchar_t*)wmemchr(__s, __c, __n);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD const wchar_t*
wmemchr(const wchar_t* __s, wchar_t __c, size_t __n) {
  return __libcpp_wmemchr(__s, __c, __n);
}
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_PREFERRED_OVERLOAD wchar_t* wmemchr(wchar_t* __s, wchar_t __c, size_t __n) {
  return __libcpp_wmemchr(__s, __c, __n);
}
}
#  endif

#  if defined(__cplusplus) && (defined(_LIBCPP_MSVCRT_LIKE) || defined(__MVS__))
extern "C" {
size_t mbsnrtowcs(
    wchar_t* __restrict __dst, const char** __restrict __src, size_t __nmc, size_t __len, mbstate_t* __restrict __ps);
size_t wcsnrtombs(
    char* __restrict __dst, const wchar_t** __restrict __src, size_t __nwc, size_t __len, mbstate_t* __restrict __ps);
} // extern "C"
#  endif // __cplusplus && (_LIBCPP_MSVCRT || __MVS__)
#endif   // _LIBCPP_HAS_WIDE_CHARACTERS
