//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <cstdalign> // deprecated in C++17, removed in C++20, but still provided by libc++ as an extension

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <cstdalign>

#ifndef __alignas_is_defined
#  error __alignas_is_defined not defined
#endif

#ifndef __alignof_is_defined
#  error __alignof_is_defined not defined
#endif

#ifdef alignas
#  error alignas should not be defined
#endif

#ifdef alignof
#  error alignof should not be defined
#endif
