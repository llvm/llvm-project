//===-- Definition of macros from float.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_FLOAT_MACROS_H
#define LLVM_LIBC_MACROS_FLOAT_MACROS_H

// Check long double.
#if defined(__linux__) && defined(__x86_64__)
#define LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80
#elif defined(__linux__) && (defined(__aarch64__) || defined(__riscv))
#define LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT128
#else
#define LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64
#endif

#ifndef FLT_RADIX
#ifdef __FLT_RADIX__
#define FLT_RADIX __FLT_RADIX__
#else
#define FLT_RADIX 2
#endif // __FLT_RADIX__
#endif // FLT_RADIX

#ifndef FLT_EVAL_METHOD
#ifdef __FLT_EVAL_METHOD__
#define FLT_EVAL_METHOD __FLT_EVAL_METHOD__
#else
#define FLT_EVAL_METHOD 0
#endif // __FLT_EVAL_METHOD__
#endif // FLT_EVAL_METHOD

#ifndef FLT_ROUNDS
#define FLT_ROUNDS 1
#endif // FLT_ROUNDS

#ifndef FLT_DECIMAL_DIG
#ifdef __FLT_DECIMAL_DIG__
#define FLT_DECIMAL_DIG __FLT_DECIMAL_DIG__
#else
#define FLT_DECIMAL_DIG 9
#endif // __FLT_DECIMAL_DIG__
#endif // FLT_DECIMAL_DIG

#ifndef DBL_DECIMAL_DIG
#ifdef __DBL_DECIMAL_DIG__
#define DBL_DECIMAL_DIG __DBL_DECIMAL_DIG__
#else
#define DBL_DECIMAL_DIG 17
#endif // __DBL_DECIMAL_DIG__
#endif // DBL_DECIMAL_DIG

#ifndef LDBL_DECIMAL_DIG
#ifdef __LDBL_DECIMAL_DIG__
#define LDBL_DECIMAL_DIG __LDBL_DECIMAL_DIG__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80)
#define LDBL_DECIMAL_DIG 21
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT128)
#define LDBL_DECIMAL_DIG 36
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64
#define LDBL_DECIMAL_DIG DBL_DECIMAL_DIG
#endif // __LDBL_DECIMAL_DIG
#endif // LDBL_DECIMAL_DIG

#ifndef DECIMAL_DIG
#ifdef __DECIMAL_DIG__
#define DECIMAL_DIG __DECIMAL_DIG__
#else
#define DECIMAL_DIG LDBL_DECIMAL_DIG
#endif // __DECIMAL_DIG
#endif // DECIMAL_DIG

#ifndef FLT_DIG
#ifdef __FLT_DIG__
#define FLT_DIG __FLT_DIG__
#else
#define FLT_DIG 6
#endif // __FLT_DIG__
#endif // FLT_DIG

#ifndef DBL_DIG
#ifdef __DBL_DIG__
#define DBL_DIG __DBL_DIG__
#else
#define DBL_DIG 15
#endif // __DBL_DIG__
#endif // DBL_DIG

#ifndef LDBL_DIG
#ifdef __LDBL_DIG__
#define LDBL_DIG __LDBL_DIG__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80)
#define LDBL_DIG 18
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT128)
#define LDBL_DIG 33
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64
#define LDBL_DIG DBL_DIG
#endif // __LDBL_DIG__
#endif // LDBL_DIG

#ifndef FLT_MANT_DIG
#ifdef __FLT_MANT_DIG__
#define FLT_MANT_DIG __FLT_MANT_DIG__
#else
#define FLT_MANT_DIG 24
#endif // __FLT_MANT_DIG__
#endif // FLT_MANT_DIG

#ifndef DBL_MANT_DIG
#ifdef __DBL_MANT_DIG__
#define DBL_MANT_DIG __DBL_MANT_DIG__
#else
#define DBL_MANT_DIG 53
#endif // __DBL_MANT_DIG__
#endif // DBL_MANT_DIG

#ifndef LDBL_MANT_DIG
#ifdef __LDBL_MANT_DIG__
#define LDBL_MANT_DIG __LDBL_MANT_DIG__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80)
#define LDBL_MANT_DIG 64
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT128)
#define LDBL_MANT_DIG 113
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64
#define LDBL_MANT_DIG DBL_MANT_DIG
#endif // __LDBL_MANT_DIG__
#endif // LDBL_MANT_DIG

#ifndef FLT_MIN
#ifdef __FLT_MIN__
#define FLT_MIN __FLT_MIN__
#else
#define FLT_MIN 0x1.0p-126f
#endif // __FLT_MIN__
#endif // FLT_MIN

#ifndef DBL_MIN
#ifdef __DBL_MIN__
#define DBL_MIN __DBL_MIN__
#else
#define DBL_MIN 0x1.0p-1022
#endif // __DBL_MIN__
#endif // DBL_MIN

#ifndef LDBL_MIN
#ifdef __LDBL_MIN__
#define LDBL_MIN __LDBL_MIN__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64)
#define LDBL_MIN 0x1.0p-1022L
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80 or FLOAT128
#define LDBL_MIN 0x1.0p-16382L
#endif // __LDBL_MIN__
#endif // LDBL_MIN

#ifndef FLT_MAX
#ifdef __FLT_MAX__
#define FLT_MAX __FLT_MAX__
#else
#define FLT_MAX 0x1.fffffep+127f
#endif // __FLT_MAX__
#endif // FLT_MAX

#ifndef DBL_MAX
#ifdef __DBL_MAX__
#define DBL_MAX __DBL_MAX__
#else
#define DBL_MAX 0x1.ffff'ffff'ffff'fp+1023
#endif // __DBL_MAX__
#endif // DBL_MAX

#ifndef LDBL_MAX
#ifdef __LDBL_MAX__
#define LDBL_MAX __LDBL_MAX__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80)
#define LDBL_MAX 0x1.ffff'ffff'ffff'fffep+16383L
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT128)
#define LDBL_MAX 0x1.ffff'ffff'ffff'ffff'ffff'ffff'ffffp+16383L
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64
#define LDBL_MAX 0x1.ffff'ffff'ffff'fp+1023L
#endif // __LDBL_MAX__
#endif // LDBL_MAX

#ifndef FLT_TRUE_MIN
#ifdef __FLT_DENORM_MIN__
#define FLT_TRUE_MIN __FLT_DENORM_MIN__
#else
#define FLT_TRUE_MIN 0x1.0p-149f
#endif // __FLT_DENORM_MIN__
#endif // FLT_TRUE_MIN

#ifndef DBL_TRUE_MIN
#ifdef __DBL_DENORM_MIN__
#define DBL_TRUE_MIN __DBL_DENORM_MIN__
#else
#define DBL_TRUE_MIN 0x1.0p-1074
#endif // __DBL_DENORM_MIN__
#endif // DBL_TRUE_MIN

#ifndef LDBL_TRUE_MIN
#ifdef __LDBL_DENORM_MIN__
#define LDBL_TRUE_MIN __LDBL_DENORM_MIN__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80)
#define LDBL_TRUE_MIN 0x1.0p-16445L
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT128)
#define LDBL_TRUE_MIN 0x1.0p-16494L
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64
#define LDBL_TRUE_MIN 0x1.0p-1074L
#endif // __LDBL_DENORM_MIN__
#endif // LDBL_TRUE_MIN

#ifndef FLT_EPSILON
#ifdef __FLT_EPSILON__
#define FLT_EPSILON __FLT_EPSILON__
#else
#define FLT_EPSILON 0x1.0p-23f
#endif // __FLT_EPSILON__
#endif // FLT_EPSILON

#ifndef DBL_EPSILON
#ifdef __DBL_EPSILON__
#define DBL_EPSILON __DBL_EPSILON__
#else
#define DBL_EPSILON 0x1.0p-52
#endif // __DBL_EPSILON__
#endif // DBL_EPSILON

#ifndef LDBL_EPSILON
#ifdef __LDBL_EPSILON__
#define LDBL_EPSILON __LDBL_EPSILON__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80)
#define LDBL_EPSILON 0x1.0p-63L
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT128)
#define LDBL_EPSILON 0x1.0p-112L
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64
#define LDBL_EPSILON 0x1.0p-52L
#endif // __LDBL_EPSILON__
#endif // LDBL_EPSILON

#ifndef FLT_MIN_EXP
#ifdef __FLT_MIN_EXP__
#define FLT_MIN_EXP __FLT_MIN_EXP__
#else
#define FLT_MIN_EXP (-125)
#endif // __FLT_MIN_EXP__
#endif // FLT_MIN_EXP

#ifndef DBL_MIN_EXP
#ifdef __DBL_MIN_EXP__
#define DBL_MIN_EXP __DBL_MIN_EXP__
#else
#define DBL_MIN_EXP (-1021)
#endif // __DBL_MIN_EXP__
#endif // DBL_MIN_EXP

#ifndef LDBL_MIN_EXP
#ifdef __LDBL_MIN_EXP__
#define LDBL_MIN_EXP __LDBL_MIN_EXP__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64)
#define LDBL_MIN_EXP DBL_MIN_EXP
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80 or FLOAT128
#define LDBL_MIN_EXP (-16381)
#endif // __LDBL_MIN_EXP__
#endif // LDBL_MIN_EXP

#ifndef FLT_MIN_10_EXP
#ifdef __FLT_MIN_10_EXP__
#define FLT_MIN_10_EXP __FLT_MIN_10_EXP__
#else
#define FLT_MIN_10_EXP (-37)
#endif // __FLT_MIN_10_EXP__
#endif // FLT_MIN_10_EXP

#ifndef DBL_MIN_10_EXP
#ifdef __DBL_MIN_10_EXP__
#define DBL_MIN_10_EXP __DBL_MIN_10_EXP__
#else
#define DBL_MIN_10_EXP (-307)
#endif // __DBL_MIN_10_EXP__
#endif // DBL_MIN_10_EXP

#ifndef LDBL_MIN_10_EXP
#ifdef __LDBL_MIN_10_EXP__
#define LDBL_MIN_10_EXP __LDBL_MIN_10_EXP__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64)
#define LDBL_MIN_10_EXP DBL_MIN_10_EXP
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80 or FLOAT128
#define LDBL_MIN_10_EXP (-4931)
#endif // __LDBL_MIN_10_EXP__
#endif // LDBL_MIN_10_EXP

#ifndef FLT_MAX_EXP
#ifdef __FLT_MAX_EXP__
#define FLT_MAX_EXP __FLT_MAX_EXP__
#else
#define FLT_MAX_EXP 128
#endif // __FLT_MAX_EXP__
#endif // FLT_MAX_EXP

#ifndef DBL_MAX_EXP
#ifdef __DBL_MAX_EXP__
#define DBL_MAX_EXP __DBL_MAX_EXP__
#else
#define DBL_MAX_EXP 1024
#endif // __DBL_MAX_EXP__
#endif // DBL_MAX_EXP

#ifndef LDBL_MAX_EXP
#ifdef __LDBL_MAX_EXP__
#define LDBL_MAX_EXP __LDBL_MAX_EXP__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64)
#define LDBL_MAX_EXP DBL_MAX_EXP
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80 or FLOAT128
#define LDBL_MAX_EXP 16384
#endif // __LDBL_MAX_EXP__
#endif // LDBL_MAX_EXP

#ifndef FLT_MAX_10_EXP
#ifdef __FLT_MAX_10_EXP__
#define FLT_MAX_10_EXP __FLT_MAX_10_EXP__
#else
#define FLT_MAX_10_EXP 38
#endif // __FLT_MAX_10_EXP__
#endif // FLT_MAX_10_EXP

#ifndef DBL_MAX_10_EXP
#ifdef __DBL_MAX_10_EXP__
#define DBL_MAX_10_EXP __DBL_MAX_10_EXP__
#else
#define DBL_MAX_10_EXP 308
#endif // __DBL_MAX_10_EXP__
#endif // DBL_MAX_10_EXP

#ifndef LDBL_MAX_10_EXP
#ifdef __LDBL_MAX_10_EXP__
#define LDBL_MAX_10_EXP __LDBL_MAX_10_EXP__
#elif defined(LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64)
#define LDBL_MAX_10_EXP DBL_MAX_10_EXP
#else // LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80 or FLOAT128
#define LDBL_MAX_10_EXP 4932
#endif // __LDBL_MAX_10_EXP__
#endif // LDBL_MAX_10_EXP

// TODO: Add FLT16 and FLT128 constants.

// Cleanup
#undef LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT64
#undef LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT80
#undef LLVM_LIBC_MACROS_LONG_DOUBLE_IS_FLOAT128

#endif // LLVM_LIBC_MACROS_FLOAT_MACROS_H
