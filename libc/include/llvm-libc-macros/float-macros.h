//===-- Definition of macros from float.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_FLOAT_MACROS_H
#define LLVM_LIBC_MACROS_FLOAT_MACROS_H

#ifdef __FLT_RADIX__
#define FLT_RADIX __FLT_RADIX__
#else
#define FLT_RADIX 2
#endif // FLT_RADIX

// #ifdef FLT_EVAL_METHOD
// #define FLT_EVAL_METHOD __FLT_EVAL_METHOD__
// #endif // FLT_EVAL_METHOD
// #define FLT_EVAL_METHOD 0
//clarifiy

#ifdef DECIMAL_DIG
#define DECIMAL_DIG __DECIMAL_DIG__
#else
#define DECIMAL_DIG 10
#endif // DECIMAL_DIG
//clarify

#ifdef FLT_DECIMAL_DIG
#define FLT_DECIMAL_DIG __FLT_DECIMAL_DIG__
#else
#define FLT_DECIMAL_DIG 9
#endif // FLT_DECIMAL_DIG

#ifdef DBL_DECIMAL_DIG
#define DBL_DECIMAL_DIG __DBL_DECIMAL_DIG__
#else
#define DBL_DECIMAL_DIG 17
#endif // DBL_DECIMAL_DIG

#ifdef LDBL_DECIMAL_DIG
#define LDBL_DECIMAL_DIG __LDBL_DECIMAL_DIG__
#else
#define LDBL_DECIMAL_DIG 10
#endif // LDBL_DECIMAL_DIG
//clarify

#ifdef FLT_DIG
#define FLT_DIG __FLT_DIG__
#else
#define FLT_DIG 6
#endif // FLT_DIG

#ifdef DBL_DIG
#define DBL_DIG __DBL_DIG__
#else
#define DBL_DIG 15
#endif // DBL_DIG

#ifdef LDBL_DIG
#define LDBL_DIG __LDBL_DIG__
#else
#define LDBL_DIG 10
#endif // LDBL_DIG
//clarify

#ifdef FLT_MANT_DIG
#define FLT_MANT_DIG __FLT_MANT_DIG__
#else
#define FLT_MANT_DIG 24
#endif // FLT_MANT_DIG

#ifdef DBL_MANT_DIG
#define DBL_MANT_DIG __DBL_MANT_DIG__
#else
#define DBL_MANT_DIG 53
#endif // DBL_MANT_DIG

#ifdef LDBL_MANT_DIG
#define LDBL_MANT_DIG __LDBL_MANT_DIG__
#else
#define LDBL_MANT_DIG 113
#endif // LDBL_MANT_DIG
//clarify

#ifdef FLT_MIN
#define FLT_MIN __FLT_MIN__
#else
#define FLT_MIN 1.17549435082228750796873653722224568e-38F
#endif // FLT_MIN

#ifdef DBL_MIN
#define DBL_MIN __DBL_MIN__
#else
#define DBL_MIN 2.22507385850720138309023271733240406e-308L
#endif // DBL_MIN

#ifdef LDBL_MIN
#define LDBL_MIN __LDBL_MIN__
#else
#define LDBL_MIN 1E-37
#endif // LDBL_MIN
//clarify

#ifdef FLT_MAX
#define FLT_MAX __FLT_MAX__
#else
#define FLT_MAX 3.40282346638528859811704183484516925e+38F
#endif // FLT_MAX

#ifdef DBL_MAX
#define DBL_MAX __DBL_MAX__
#else
#define DBL_MAX 1.79769313486231570814527423731704357e+308L
#endif // DBL_MAX

#ifdef LDBL_MAX
#define LDBL_MAX __LDBL_MAX__
#else
#define LDBL_MAX 1E+37
#endif // LDBL_MAX
//clarify

#ifdef FLT_TRUE_MIN
#define FLT_TRUE_MIN __FLT_DNORM_MIN 
#else 
#define FLT_TRUE_MIN 1.40129846432481707092372958328991613e-45F 
#endif // FLT_TRUE_MIN

#ifdef DBL_TRUE_MIN
#define DBL_TRUE_MIN __DBL_DENORM_MIN__ 
#else 
#define DBL_TRUE_MIN 4.94065645841246544176568792868221372e-324L 
#endif // DBL_TRUE_MIN

#ifdef LDBL_TRUE_MIN
#define LDBL_TRUE_MIN __LDBL_DENORM_MIN__ 
#else 
#define LDBL_TRUE_MIN 1E-37 
#endif // LDBL_TRUE_MIN
//clarify

#ifdef FLT_EPSILON
#define FLT_EPSILON __FLT_EPSILON__
#else
#define FLT_EPSILON 1.1920928955078125e-7F
#endif // FLT_EPSILON

#ifdef DBL_EPSILON
#define DBL_EPSILON __DBL_EPSILON__
#else
#define DBL_EPSILON 2.22044604925031308084726333618164062e-16L 
#endif // DBL_EPSILON

#ifdef LDBL_EPSILON
#define LDBL_EPSILON __LDBL_EPSILON__
#else
#define LDBL_EPSILON 1E-9
#endif // LDBL_EPSILON
//clarify

#ifdef FLT_MIN_EXP
#define FLT_MIN_EXP __FLT_MIN_EXP__
#else 
#define FLT_MIN_EXP (-125)
#endif // FLT_MIN_EXP

#ifdef DBL_MIN_EXP
#define DBL_MIN_EXP __DBL_MIN_EXP__
#else
#define DBL_MIN_EXP (-1021)
#endif // DBL_MIN_EXP

#ifdef LDBL_MIN_EXP
#define LDBL_MIN_EXP __LDBL_MIN_EXP__
#endif // LDBL_MIN_EXP
//clarify

#ifdef FLT_MIN_10_EXP
#define FLT_MIN_10_EXP __FLT_MIN_10_EXP__
#else
#define FLT_MIN_10_EXP (-37)
#endif // FLT_MIN_10_EXP

#ifdef DBL_MIN_10_EXP
#define DBL_MIN_10_EXP __DBL_MIN_10_EXP__
#else
#define DBL_MIN_10_EXP (-307)
#endif // DBL_MIN_10_EXP

#ifdef LDBL_MIN_10_EXP
#define LDBL_MIN_10_EXP __LDBL_MIN_10_EXP__
#else
#define LDBL_MIN_10_EXP -37
#endif // LDBL_MIN_10_EXP
//clarify

#ifdef FLT_MAX_EXP
#define FLT_MAX_EXP __FLT_MAX_EXP__
#else
#define FLT_MAX_EXP 128
#endif // FLT_MAX_EXP

#ifdef DBL_MAX_EXP
#define DBL_MAX_EXP __DBL_MAX_EXP__
#else
#define DBL_MAX_EXP 1024
#endif // DBL_MAX_EXP

#ifdef LDBL_MAX_EXP
#define LDBL_MAX_EXP __LDBL_MAX_EXP__
#endif // LDBL_MAX_EXP
//clarify

#ifdef FLT_MAX_10_EXP
#define FLT_MAX_10_EXP __FLT_MAX_10_EXP__
#else
#define FLT_MAX_10_EXP 38
#endif // FLT_MAX_10_EXP

#ifdef DBL_MAX_10_EXP
#define DBL_MAX_10_EXP __DBL_MAX_10_EXP__
#else
#define DBL_MAX_10_EXP 308
#endif // DBL_MAX_10_EXP

#ifdef LDBL_MAX_10_EXP
#define LDBL_MAX_10_EXP __LDBL_MAX_10_EXP__
#else
#define LDBL_MAX_10_EXP +37
#endif // LDBL_MAX_10_EXP

#if defined(LIBC_TARGET_ARCH_IS_RISCV64) || defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)

#endif

#if defined(LIBC_TARGET_ARCH_IS_X86_32)

#endif

#if defined(LIBC_TARGET_ARCH_IS_X86_64)

#endif

// TODO: Add FLT16 and FLT128 constants.

#endif // LLVM_LIBC_MACROS_FLOAT_MACROS_H
