//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_convertible

// Test the fallback implementation.

// libc++ provides a fallback implementation of the compiler trait
// `__is_convertible` with the same name when clang doesn't.
// Because this test forces the use of the fallback even when clang provides
// it causing a keyword incompatibility.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wkeyword-compat"
#endif

#define _LIBCPP_USE_IS_CONVERTIBLE_FALLBACK
#include "is_convertible.pass.cpp"

#include "test_macros.h"

