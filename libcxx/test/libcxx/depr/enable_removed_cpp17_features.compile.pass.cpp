//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that defining _LIBCPP_ENABLE_CXX17_REMOVED_FEATURES correctly defines
// _LIBCPP_ENABLE_CXX17_REMOVED_FOO for each individual component macro.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES

#include <__config>

#include "test_macros.h"

#ifndef _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR
#  error _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR must be defined
#endif

#ifndef _LIBCPP_ENABLE_CXX17_REMOVED_BINDERS
#  error _LIBCPP_ENABLE_CXX17_REMOVED_BINDERS must be defined
#endif

#ifndef _LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE
#  error _LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE must be defined
#endif

#ifndef _LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS
#error _LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS must be defined
#endif

#ifndef _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR
#error _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR must be defined
#endif
