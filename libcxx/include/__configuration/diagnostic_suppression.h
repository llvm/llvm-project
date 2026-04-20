// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_DIAGNOSTIC_SUPPRESSION_H
#define _LIBCPP___CONFIGURATION_DIAGNOSTIC_SUPPRESSION_H

#include <__config_site>
#include <__configuration/compiler.h>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

#ifdef _LIBCPP_COMPILER_CLANG_BASED
#  define _LIBCPP_DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#  define _LIBCPP_DIAGNOSTIC_POP _Pragma("clang diagnostic pop")
#  define _LIBCPP_CLANG_DIAGNOSTIC_IGNORED(str) _Pragma(_LIBCPP_TOSTRING(clang diagnostic ignored str))
#  define _LIBCPP_GCC_DIAGNOSTIC_IGNORED(str)
#elif defined(_LIBCPP_COMPILER_GCC)
#  define _LIBCPP_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#  define _LIBCPP_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#  define _LIBCPP_CLANG_DIAGNOSTIC_IGNORED(str)
#  define _LIBCPP_GCC_DIAGNOSTIC_IGNORED(str) _Pragma(_LIBCPP_TOSTRING(GCC diagnostic ignored str))
#else
#  define _LIBCPP_DIAGNOSTIC_PUSH
#  define _LIBCPP_DIAGNOSTIC_POP
#  define _LIBCPP_CLANG_DIAGNOSTIC_IGNORED(str)
#  define _LIBCPP_GCC_DIAGNOSTIC_IGNORED(str)
#endif

// Macros to enter and leave a state where deprecation warnings are suppressed.
#define _LIBCPP_SUPPRESS_DEPRECATED_PUSH                                                                               \
  _LIBCPP_DIAGNOSTIC_PUSH _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wdeprecated")                                             \
      _LIBCPP_GCC_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
#define _LIBCPP_SUPPRESS_DEPRECATED_POP _LIBCPP_DIAGNOSTIC_POP

#endif // _LIBCPP___CONFIGURATION_DIAGNOSTIC_SUPPRESSION_H
