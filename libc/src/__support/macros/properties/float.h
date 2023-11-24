//===-- Float type support --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Floating point properties are a combination of compiler support, target OS
// and target architecture.

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_FLOAT_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_FLOAT_H

#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/compiler.h"
#include "src/__support/macros/properties/os.h"

// https://developer.arm.com/documentation/dui0491/i/C-and-C---Implementation-Details/Basic-data-types
// https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
// https://docs.amd.com/bundle/HIP-Programming-Guide-v5.1/page/Programming_with_HIP.html
#if defined(LIBC_TARGET_OS_IS_WINDOWS) ||                                      \
    (defined(LIBC_TARGET_OS_IS_MACOS) &&                                       \
     defined(LIBC_TARGET_ARCH_IS_AARCH64)) ||                                  \
    defined(LIBC_TARGET_ARCH_IS_ARM) || defined(LIBC_TARGET_ARCH_IS_NVPTX) ||  \
    defined(LIBC_TARGET_ARCH_IS_AMDGPU)
#define LONG_DOUBLE_IS_DOUBLE
#endif

#if !defined(LONG_DOUBLE_IS_DOUBLE) && defined(LIBC_TARGET_ARCH_IS_X86)
#define SPECIAL_X86_LONG_DOUBLE
#endif

// Check compiler features
#if defined(FLT128_MANT_DIG)
#define LIBC_COMPILER_HAS_FLOAT128
using float128 = _Float128;
#elif defined(__SIZEOF_FLOAT128__)
#define LIBC_COMPILER_HAS_FLOAT128
using float128 = __float128;
#elif (defined(__linux__) && defined(__aarch64__))
#define LIBC_COMPILER_HAS_FLOAT128
#define LIBC_FLOAT128_IS_LONG_DOUBLE
using float128 = long double;
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_FLOAT_H
