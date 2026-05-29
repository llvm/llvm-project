// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_UTILITY_H
#define _LIBCPP___CONFIGURATION_UTILITY_H

#include <__config_site>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

#define _LIBCPP_TOSTRING2(x) #x
#define _LIBCPP_TOSTRING(x) _LIBCPP_TOSTRING2(x)

#endif // _LIBCPP___CONFIGURATION_UTILITY_H
