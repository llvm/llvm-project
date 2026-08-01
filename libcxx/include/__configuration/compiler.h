// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_COMPILER_H
#define _LIBCPP___CONFIGURATION_COMPILER_H

#include <__config_site>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

#if defined(__apple_build_version__)
// Given AppleClang XX.Y.Z, _LIBCPP_APPLE_CLANG_VER is XXYZ (e.g. AppleClang 14.0.3 => 1403)
#  define _LIBCPP_COMPILER_CLANG_BASED
#  define _LIBCPP_APPLE_CLANG_VER (__apple_build_version__ / 10000)
#elif defined(__clang__)
#  define _LIBCPP_COMPILER_CLANG_BASED
#  define _LIBCPP_CLANG_VER (__clang_major__ * 100 + __clang_minor__)
#elif defined(__GNUC__)
#  define _LIBCPP_COMPILER_GCC
#  define _LIBCPP_GCC_VER (__GNUC__ * 100 + __GNUC_MINOR__)
#endif

#ifdef __cplusplus

// Warn if a compiler version is used that is not supported anymore
// LLVM RELEASE Update the minimum compiler versions
#  if defined(_LIBCPP_CLANG_VER)
#    if _LIBCPP_CLANG_VER < 2101
#      warning "Libc++ only supports Clang 21 and later"
#    endif
#  elif defined(_LIBCPP_APPLE_CLANG_VER)
#    if _LIBCPP_APPLE_CLANG_VER < 2100
#      warning "Libc++ only supports AppleClang 26.4 and later"
#    endif
#  elif defined(_LIBCPP_GCC_VER)
#    if _LIBCPP_GCC_VER < 1500
#      warning "Libc++ only supports GCC 15 and later"
#    endif
#  endif

#  ifndef __has_constexpr_builtin
#    define __has_constexpr_builtin(x) 0
#  endif

// This checks wheter a Clang module is built
#  ifndef __building_module
#    define __building_module(...) 0
#  endif

// '__is_identifier' returns '0' if '__x' is a reserved identifier provided by
// the compiler and '1' otherwise.
#  ifndef __is_identifier
#    define __is_identifier(__x) 1
#  endif

#  define __has_keyword(__x) !(__is_identifier(__x))

#  ifndef __has_warning
#    define __has_warning(...) 0
#  endif

#  if !defined(_LIBCPP_COMPILER_CLANG_BASED) && __cplusplus < 201103L
#    error "libc++ only supports C++03 with Clang-based compilers. Please enable C++11"
#  endif

#endif

#endif // _LIBCPP___CONFIGURATION_COMPILER_H
