// -*- C++ -*-
//===------------------- support/android/locale_bionic.h ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_ANDROID_LOCALE_BIONIC_H
#define _LIBCPP_SUPPORT_ANDROID_LOCALE_BIONIC_H

#if defined(__BIONIC__)

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <xlocale.h>

#ifdef __cplusplus
}
#endif

#if defined(__ANDROID__)

#include <android/api-level.h>

// Android gained most locale aware functions in L (API level 21)
#if __ANDROID_API__ < 21
#include <support/xlocale/__posix_l_fallback.h>
#endif

// The strto* family was added in O (API Level 26)
#if __ANDROID_API__ < 26

#if defined(__cplusplus)
extern "C" {
#endif

inline _LIBCPP_ALWAYS_INLINE float strtof_l(const char* __nptr, char** __endptr,
                                            locale_t) {
  return ::strtof(__nptr, __endptr);
}

inline _LIBCPP_ALWAYS_INLINE double strtod_l(const char* __nptr,
                                             char** __endptr, locale_t) {
  return ::strtod(__nptr, __endptr);
}

inline _LIBCPP_ALWAYS_INLINE long strtol_l(const char* __nptr, char** __endptr,
                                           int __base, locale_t) {
  return ::strtol(__nptr, __endptr, __base);
}

#if defined(__cplusplus)
}
#endif

#endif // __ANDROID_API__ < 26

#endif // defined(__ANDROID__)

#endif // defined(__BIONIC__)
#endif // _LIBCPP_SUPPORT_ANDROID_LOCALE_BIONIC_H
