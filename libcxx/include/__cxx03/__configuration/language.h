// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___CONFIGURATION_LANGUAGE_H
#define _LIBCPP___CXX03___CONFIGURATION_LANGUAGE_H

#include <__cxx03/__configuration/config_site_shim.h>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

#if !defined(__cpp_rtti) || __cpp_rtti < 199711L
#  define _LIBCPP_HAS_NO_RTTI
#endif

#if !defined(__cpp_exceptions) || __cpp_exceptions < 199711L
#  define _LIBCPP_HAS_NO_EXCEPTIONS
#endif

#endif // _LIBCPP___CXX03___CONFIGURATION_LANGUAGE_H
