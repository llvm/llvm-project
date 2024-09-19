//===-- include/flang/Runtime/matmul-instances.inc --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helper macros to instantiate MATMUL/MATMUL_TRANSPOSE definitions
// for different data types of the input arguments.
//===----------------------------------------------------------------------===//

#ifndef MATMUL_INSTANCE
#error "Define MATMUL_INSTANCE before including this file"
#endif

#ifndef MATMUL_DIRECT_INSTANCE
#error "Define MATMUL_DIRECT_INSTANCE before including this file"
#endif

#ifndef MATMUL_FORCE_ALL_TYPES
#error "Define MATMUL_FORCE_ALL_TYPES to 0 or 1 before including this file"
#endif

// clang-format off

#define FOREACH_MATMUL_TYPE_PAIR(macro)         \
  macro(Integer, 1, Integer, 1)                 \
  macro(Integer, 1, Integer, 2)                 \
  macro(Integer, 1, Integer, 4)                 \
  macro(Integer, 1, Integer, 8)                 \
  macro(Integer, 2, Integer, 1)                 \
  macro(Integer, 2, Integer, 2)                 \
  macro(Integer, 2, Integer, 4)                 \
  macro(Integer, 2, Integer, 8)                 \
  macro(Integer, 4, Integer, 1)                 \
  macro(Integer, 4, Integer, 2)                 \
  macro(Integer, 4, Integer, 4)                 \
  macro(Integer, 4, Integer, 8)                 \
  macro(Integer, 8, Integer, 1)                 \
  macro(Integer, 8, Integer, 2)                 \
  macro(Integer, 8, Integer, 4)                 \
  macro(Integer, 8, Integer, 8)                 \
  macro(Integer, 1, Real, 4)                    \
  macro(Integer, 1, Real, 8)                    \
  macro(Integer, 2, Real, 4)                    \
  macro(Integer, 2, Real, 8)                    \
  macro(Integer, 4, Real, 4)                    \
  macro(Integer, 4, Real, 8)                    \
  macro(Integer, 8, Real, 4)                    \
  macro(Integer, 8, Real, 8)                    \
  macro(Integer, 1, Complex, 4)                 \
  macro(Integer, 1, Complex, 8)                 \
  macro(Integer, 2, Complex, 4)                 \
  macro(Integer, 2, Complex, 8)                 \
  macro(Integer, 4, Complex, 4)                 \
  macro(Integer, 4, Complex, 8)                 \
  macro(Integer, 8, Complex, 4)                 \
  macro(Integer, 8, Complex, 8)                 \
  macro(Real, 4, Integer, 1)                    \
  macro(Real, 4, Integer, 2)                    \
  macro(Real, 4, Integer, 4)                    \
  macro(Real, 4, Integer, 8)                    \
  macro(Real, 8, Integer, 1)                    \
  macro(Real, 8, Integer, 2)                    \
  macro(Real, 8, Integer, 4)                    \
  macro(Real, 8, Integer, 8)                    \
  macro(Real, 4, Real, 4)                       \
  macro(Real, 4, Real, 8)                       \
  macro(Real, 8, Real, 4)                       \
  macro(Real, 8, Real, 8)                       \
  macro(Real, 4, Complex, 4)                    \
  macro(Real, 4, Complex, 8)                    \
  macro(Real, 8, Complex, 4)                    \
  macro(Real, 8, Complex, 8)                    \
  macro(Complex, 4, Integer, 1)                 \
  macro(Complex, 4, Integer, 2)                 \
  macro(Complex, 4, Integer, 4)                 \
  macro(Complex, 4, Integer, 8)                 \
  macro(Complex, 8, Integer, 1)                 \
  macro(Complex, 8, Integer, 2)                 \
  macro(Complex, 8, Integer, 4)                 \
  macro(Complex, 8, Integer, 8)                 \
  macro(Complex, 4, Real, 4)                    \
  macro(Complex, 4, Real, 8)                    \
  macro(Complex, 8, Real, 4)                    \
  macro(Complex, 8, Real, 8)                    \
  macro(Complex, 4, Complex, 4)                 \
  macro(Complex, 4, Complex, 8)                 \
  macro(Complex, 8, Complex, 4)                 \
  macro(Complex, 8, Complex, 8)                 \

FOREACH_MATMUL_TYPE_PAIR(MATMUL_INSTANCE)
FOREACH_MATMUL_TYPE_PAIR(MATMUL_DIRECT_INSTANCE)

#if MATMUL_FORCE_ALL_TYPES || (defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T)
#define FOREACH_MATMUL_TYPE_PAIR_WITH_INT16(macro)      \
  macro(Integer, 16, Integer, 1)                        \
  macro(Integer, 16, Integer, 2)                        \
  macro(Integer, 16, Integer, 4)                        \
  macro(Integer, 16, Integer, 8)                        \
  macro(Integer, 16, Integer, 16)                       \
  macro(Integer, 16, Real, 4)                           \
  macro(Integer, 16, Real, 8)                           \
  macro(Integer, 16, Complex, 4)                        \
  macro(Integer, 16, Complex, 8)                        \
  macro(Real, 4, Integer, 16)                           \
  macro(Real, 8, Integer, 16)                           \
  macro(Complex, 4, Integer, 16)                        \
  macro(Complex, 8, Integer, 16)                        \

FOREACH_MATMUL_TYPE_PAIR_WITH_INT16(MATMUL_INSTANCE)
FOREACH_MATMUL_TYPE_PAIR_WITH_INT16(MATMUL_DIRECT_INSTANCE)

#if MATMUL_FORCE_ALL_TYPES || HAS_FLOAT80
MATMUL_INSTANCE(Integer, 16, Real, 10)
MATMUL_INSTANCE(Integer, 16, Complex, 10)
MATMUL_INSTANCE(Real, 10, Integer, 16)
MATMUL_INSTANCE(Complex, 10, Integer, 16)
MATMUL_DIRECT_INSTANCE(Integer, 16, Real, 10)
MATMUL_DIRECT_INSTANCE(Integer, 16, Complex, 10)
MATMUL_DIRECT_INSTANCE(Real, 10, Integer, 16)
MATMUL_DIRECT_INSTANCE(Complex, 10, Integer, 16)
#endif
#if MATMUL_FORCE_ALL_TYPES || (LDBL_MANT_DIG == 113 || HAS_FLOAT128)
MATMUL_INSTANCE(Integer, 16, Real, 16)
MATMUL_INSTANCE(Integer, 16, Complex, 16)
MATMUL_INSTANCE(Real, 16, Integer, 16)
MATMUL_INSTANCE(Complex, 16, Integer, 16)
MATMUL_DIRECT_INSTANCE(Integer, 16, Real, 16)
MATMUL_DIRECT_INSTANCE(Integer, 16, Complex, 16)
MATMUL_DIRECT_INSTANCE(Real, 16, Integer, 16)
MATMUL_DIRECT_INSTANCE(Complex, 16, Integer, 16)
#endif
#endif // MATMUL_FORCE_ALL_TYPES || (defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T)

#if MATMUL_FORCE_ALL_TYPES || HAS_FLOAT80
#define FOREACH_MATMUL_TYPE_PAIR_WITH_REAL10(macro)         \
  macro(Integer, 1, Real, 10)                               \
  macro(Integer, 1, Complex, 10)                            \
  macro(Integer, 2, Real, 10)                               \
  macro(Integer, 2, Complex, 10)                            \
  macro(Integer, 4, Real, 10)                               \
  macro(Integer, 4, Complex, 10)                            \
  macro(Integer, 8, Real, 10)                               \
  macro(Integer, 8, Complex, 10)                            \
  macro(Real, 4, Real, 10)                                  \
  macro(Real, 4, Complex, 10)                               \
  macro(Real, 8, Real, 10)                                  \
  macro(Real, 8, Complex, 10)                               \
  macro(Real, 10, Integer, 1)                               \
  macro(Real, 10, Integer, 2)                               \
  macro(Real, 10, Integer, 4)                               \
  macro(Real, 10, Integer, 8)                               \
  macro(Real, 10, Real, 4)                                  \
  macro(Real, 10, Real, 8)                                  \
  macro(Real, 10, Real, 10)                                 \
  macro(Real, 10, Complex, 4)                               \
  macro(Real, 10, Complex, 8)                               \
  macro(Real, 10, Complex, 10)                              \
  macro(Complex, 4, Real, 10)                               \
  macro(Complex, 4, Complex, 10)                            \
  macro(Complex, 8, Real, 10)                               \
  macro(Complex, 8, Complex, 10)                            \
  macro(Complex, 10, Integer, 1)                            \
  macro(Complex, 10, Integer, 2)                            \
  macro(Complex, 10, Integer, 4)                            \
  macro(Complex, 10, Integer, 8)                            \
  macro(Complex, 10, Real, 4)                               \
  macro(Complex, 10, Real, 8)                               \
  macro(Complex, 10, Real, 10)                              \
  macro(Complex, 10, Complex, 4)                            \
  macro(Complex, 10, Complex, 8)                            \
  macro(Complex, 10, Complex, 10)                           \

FOREACH_MATMUL_TYPE_PAIR_WITH_REAL10(MATMUL_INSTANCE)
FOREACH_MATMUL_TYPE_PAIR_WITH_REAL10(MATMUL_DIRECT_INSTANCE)

#if MATMUL_FORCE_ALL_TYPES || HAS_FLOAT128
MATMUL_INSTANCE(Real, 10, Real, 16)
MATMUL_INSTANCE(Real, 10, Complex, 16)
MATMUL_INSTANCE(Real, 16, Real, 10)
MATMUL_INSTANCE(Real, 16, Complex, 10)
MATMUL_INSTANCE(Complex, 10, Real, 16)
MATMUL_INSTANCE(Complex, 10, Complex, 16)
MATMUL_INSTANCE(Complex, 16, Real, 10)
MATMUL_INSTANCE(Complex, 16, Complex, 10)
MATMUL_DIRECT_INSTANCE(Real, 10, Real, 16)
MATMUL_DIRECT_INSTANCE(Real, 10, Complex, 16)
MATMUL_DIRECT_INSTANCE(Real, 16, Real, 10)
MATMUL_DIRECT_INSTANCE(Real, 16, Complex, 10)
MATMUL_DIRECT_INSTANCE(Complex, 10, Real, 16)
MATMUL_DIRECT_INSTANCE(Complex, 10, Complex, 16)
MATMUL_DIRECT_INSTANCE(Complex, 16, Real, 10)
MATMUL_DIRECT_INSTANCE(Complex, 16, Complex, 10)
#endif
#endif // MATMUL_FORCE_ALL_TYPES || HAS_FLOAT80

#if MATMUL_FORCE_ALL_TYPES || (LDBL_MANT_DIG == 113 || HAS_FLOAT128)
#define FOREACH_MATMUL_TYPE_PAIR_WITH_REAL16(macro)         \
  macro(Integer, 1, Real, 16)                               \
  macro(Integer, 1, Complex, 16)                            \
  macro(Integer, 2, Real, 16)                               \
  macro(Integer, 2, Complex, 16)                            \
  macro(Integer, 4, Real, 16)                               \
  macro(Integer, 4, Complex, 16)                            \
  macro(Integer, 8, Real, 16)                               \
  macro(Integer, 8, Complex, 16)                            \
  macro(Real, 4, Real, 16)                                  \
  macro(Real, 4, Complex, 16)                               \
  macro(Real, 8, Real, 16)                                  \
  macro(Real, 8, Complex, 16)                               \
  macro(Real, 16, Integer, 1)                               \
  macro(Real, 16, Integer, 2)                               \
  macro(Real, 16, Integer, 4)                               \
  macro(Real, 16, Integer, 8)                               \
  macro(Real, 16, Real, 4)                                  \
  macro(Real, 16, Real, 8)                                  \
  macro(Real, 16, Real, 16)                                 \
  macro(Real, 16, Complex, 4)                               \
  macro(Real, 16, Complex, 8)                               \
  macro(Real, 16, Complex, 16)                              \
  macro(Complex, 4, Real, 16)                               \
  macro(Complex, 4, Complex, 16)                            \
  macro(Complex, 8, Real, 16)                               \
  macro(Complex, 8, Complex, 16)                            \
  macro(Complex, 16, Integer, 1)                            \
  macro(Complex, 16, Integer, 2)                            \
  macro(Complex, 16, Integer, 4)                            \
  macro(Complex, 16, Integer, 8)                            \
  macro(Complex, 16, Real, 4)                               \
  macro(Complex, 16, Real, 8)                               \
  macro(Complex, 16, Real, 16)                              \
  macro(Complex, 16, Complex, 4)                            \
  macro(Complex, 16, Complex, 8)                            \
  macro(Complex, 16, Complex, 16)                           \

FOREACH_MATMUL_TYPE_PAIR_WITH_REAL16(MATMUL_INSTANCE)
FOREACH_MATMUL_TYPE_PAIR_WITH_REAL16(MATMUL_DIRECT_INSTANCE)
#endif // MATMUL_FORCE_ALL_TYPES || (LDBL_MANT_DIG == 113 || HAS_FLOAT128)

#define FOREACH_MATMUL_LOGICAL_TYPE_PAIR(macro) \
  macro(Logical, 1, Logical, 1)                 \
  macro(Logical, 1, Logical, 2)                 \
  macro(Logical, 1, Logical, 4)                 \
  macro(Logical, 1, Logical, 8)                 \
  macro(Logical, 2, Logical, 1)                 \
  macro(Logical, 2, Logical, 2)                 \
  macro(Logical, 2, Logical, 4)                 \
  macro(Logical, 2, Logical, 8)                 \
  macro(Logical, 4, Logical, 1)                 \
  macro(Logical, 4, Logical, 2)                 \
  macro(Logical, 4, Logical, 4)                 \
  macro(Logical, 4, Logical, 8)                 \
  macro(Logical, 8, Logical, 1)                 \
  macro(Logical, 8, Logical, 2)                 \
  macro(Logical, 8, Logical, 4)                 \
  macro(Logical, 8, Logical, 8)                 \

FOREACH_MATMUL_LOGICAL_TYPE_PAIR(MATMUL_INSTANCE)
FOREACH_MATMUL_LOGICAL_TYPE_PAIR(MATMUL_DIRECT_INSTANCE)

#undef MATMUL_INSTANCE
#undef MATMUL_DIRECT_INSTANCE
#undef MATMUL_FORCE_ALL_TYPES

// clang-format on
