//===-- Platform specific macro definitions ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_PLATFORM_DEFS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_PLATFORM_DEFS_H

#include "src/__support/architectures.h"

#if defined(LLVM_LIBC_ARCH_X86)
#define X87_FPU
#endif

// https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
#if defined(_WIN32) || (defined(__APPLE__) && defined(__aarch64__))
#define LONG_DOUBLE_IS_DOUBLE
#endif

#if !defined(LONG_DOUBLE_IS_DOUBLE) && defined(X87_FPU)
#define SPECIAL_X86_LONG_DOUBLE
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_PLATFORM_DEFS_H
