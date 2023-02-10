//===-- Portable optimization macros ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header file defines portable macros for performance optimization.

#ifndef LLVM_LIBC_SRC_SUPPORT_MACROS_OPTIMIZATION_H
#define LLVM_LIBC_SRC_SUPPORT_MACROS_OPTIMIZATION_H

#include "src/__support/macros/properties/compiler.h"

#if defined(LIBC_COMPILER_IS_CLANG)
#define LIBC_LOOP_NOUNROLL _Pragma("nounroll")
#elif defined(LIBC_COMPILER_IS_GCC)
#define LIBC_LOOP_NOUNROLL _Pragma("GCC unroll 0")
#else
#define LIBC_LOOP_NOUNROLL
#endif

#endif /* LLVM_LIBC_SRC_SUPPORT_MACROS_OPTIMIZATION_H */
