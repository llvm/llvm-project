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

// Helpers for declaring vector constants containing all lanes the same, in a
// way that varies between little- and big-endian AArch64. Use as follows:
//
//  - define a type for the constant using V2_SPLAT_TYPE or V4_SPLAT_TYPE, with
//    a type-prefix parameter like `float64` or `int32` which the macro will
//    extend into a full scalar or vector type name.
//
//  - define the constant using V2_SPLAT_INITIALIZER or V4_SPLAT_INITIALIZER
//
//  - to get the actual vector, use MAKE_SPLAT_VECTOR(constant, suffix), where
//    'suffix' is something like `f64` or `u32` which appears in the name of a
//    NEON intrinsic to specify its element type.

#if __LITTLE_ENDIAN__

// Use the gcc language extension of defining a vector using an initializer
// list, which allows the whole vector to be statically defined in const data.
#define V2_SPLAT_TYPE(TYPE_PREFIX) TYPE_PREFIX##x2_t
#define V2_SPLAT_INITIALIZER(LANE) {LANE, LANE}
#define V4_SPLAT_TYPE(TYPE_PREFIX) TYPE_PREFIX##x4_t
#define V4_SPLAT_INITIALIZER(LANE) {LANE, LANE, LANE, LANE}
#define MAKE_SPLAT_VECTOR(VEC, FN_SUFFIX) VEC

#else

// Big-endian, that gcc language extension provokes a compiler diagnostic, so
// instead we store just one copy of the data to be duplicated across lanes,
// and perform the duplication using vdupq when loading it.
#define V2_SPLAT_TYPE(TYPE_PREFIX) TYPE_PREFIX##_t
#define V2_SPLAT_INITIALIZER(LANE) LANE
#define V4_SPLAT_TYPE(TYPE_PREFIX) TYPE_PREFIX##_t
#define V4_SPLAT_INITIALIZER(LANE) LANE
#define MAKE_SPLAT_VECTOR(LANE, FN_SUFFIX) vdupq_n_##FN_SUFFIX(LANE)

#endif
