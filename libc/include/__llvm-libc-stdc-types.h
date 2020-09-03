//===-- Definitions of common types from the C standard. ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header file does not have a header guard. It is internal to LLVM libc
// and intended to be used to pick specific definitions without polluting the
// public headers with unnecessary definitions.

#undef __LLVM_LIBC_FLOAT_T
#undef __LLVM_LIBC_DOUBLE_T

#if !defined(__FLT_EVAL_METHOD__) || __FLT_EVAL_METHOD__ == 0
#define __LLVM_LIBC_FLOAT_T float
#define __LLVM_LIBC_DOUBLE_T double
#elif __FLT_EVAL_METHOD__ == 1
#define __LLVM_LIBC_FLOAT_T double
#define __LLVM_LIBC_DOUBLE_T double
#elif __FLT_EVAL_METHOD__ == 2
#define __LLVM_LIBC_FLOAT_T long double
#define __LLVM_LIBC_DOUBLE_T long double
#else
#error "Unsupported __FLT_EVAL_METHOD__ value."
#endif

#if defined(__need_float_t) && !defined(__llvm_libc_float_t_defined)
typedef __LLVM_LIBC_FLOAT_T float_t;
#define __llvm_libc_float_t_defined
#endif // __need_float_t

#if defined(__need_double_t) && !defined(__llvm_libc_double_t_defined)
typedef __LLVM_LIBC_DOUBLE_T double_t;
#define __llvm_libc_double_t_defined
#endif // __need_double_t
