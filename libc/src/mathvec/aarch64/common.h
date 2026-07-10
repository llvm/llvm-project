//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains common utilities for AArch64 optimized mathvec functions.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/simd.h"

// Type aliases for AdvSIMD vectors.
using AdvSIMDFP32Vector = LIBC_NAMESPACE::cpp::simd<float, 4>;
using AdvSIMDFP64Vector = LIBC_NAMESPACE::cpp::simd<double, 2>;

// Returns the ptr, but hides its value from the compiler so accesses through it
// cannot be optimized based on the contents.
#define PTR_BARRIER(ptr)                                                       \
  ({                                                                           \
    decltype(ptr) __ptr = (ptr);                                               \
    asm("" : "+r"(__ptr));                                                     \
    __ptr;                                                                     \
  })

// Helpers for declaring vector constants.
#define V2(X) {X, X}
#define V4(X) {X, X, X, X}
