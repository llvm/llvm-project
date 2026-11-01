//===-- lib/fp_libc_config.h - LLVM-libc compile config --------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compile-time configuration consumed by LLVM-libc's fputil headers when they
// are included from compiler-rt builtins under COMPILER_RT_USE_LIBC_MATH.
// Pins the libc namespace to __llvm_libc_rt to avoid clashing with libc's own
// symbols, and disables errno / FP-exception bookkeeping inside libc routines
// called from builtins.
//
//===----------------------------------------------------------------------===//

#ifndef FP_LIBC_CONFIG_H
#define FP_LIBC_CONFIG_H

#ifndef LIBC_NAMESPACE
#define LIBC_NAMESPACE __llvm_libc_rt
#endif

#define LIBC_MATH (LIBC_MATH_NO_ERRNO | LIBC_MATH_NO_EXCEPT)

#endif // FP_LIBC_CONFIG_H
