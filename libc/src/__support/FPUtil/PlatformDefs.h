//===-- Platform specific macro definitions ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_PLATFORM_DEFS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_PLATFORM_DEFS_H

#if defined(__x86_64__) || defined(__i386__)
#define X87_FPU
#endif

#if defined(_WIN32)
#define LONG_DOUBLE_IS_DOUBLE
#endif

#if !defined(LONG_DOUBLE_IS_DOUBLE) && defined(X87_FPU)
#define SPECIAL_X86_LONG_DOUBLE
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_PLATFORM_DEFS_H
