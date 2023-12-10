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
#include "src/__support/macros/properties/cpu_features.h"
#include "src/__support/macros/properties/os.h"

#include <float.h> // LDBL_MANT_DIG

// 'long double' properties.
#if (LDBL_MANT_DIG == 53)
#define LIBC_LONG_DOUBLE_IS_FLOAT64
#elif (LDBL_MANT_DIG == 64)
#define LIBC_LONG_DOUBLE_IS_X86_FLOAT80
#elif (LDBL_MANT_DIG == 113)
#define LIBC_LONG_DOUBLE_IS_FLOAT128
#endif

// float16 support.
#if defined(LIBC_TARGET_ARCH_IS_X86_64) && defined(LIBC_TARGET_CPU_HAS_SSE2)
#if (defined(LIBC_COMPILER_CLANG_VER) && (LIBC_COMPILER_CLANG_VER >= 1500)) || \
    (defined(LIBC_COMPILER_GCC_VER) && (LIBC_COMPILER_GCC_VER >= 1201))
#define LIBC_COMPILER_HAS_C23_FLOAT16
#endif
#endif
#if defined(LIBC_TARGET_ARCH_IS_AARCH64)
#if (defined(LIBC_COMPILER_CLANG_VER) && (LIBC_COMPILER_CLANG_VER >= 900)) ||  \
    (defined(LIBC_COMPILER_GCC_VER) && (LIBC_COMPILER_GCC_VER >= 1301))
#define LIBC_COMPILER_HAS_C23_FLOAT16
#endif
#endif
#if defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#if (defined(LIBC_COMPILER_CLANG_VER) && (LIBC_COMPILER_CLANG_VER >= 1300)) || \
    (defined(LIBC_COMPILER_GCC_VER) && (LIBC_COMPILER_GCC_VER >= 1301))
#define LIBC_COMPILER_HAS_C23_FLOAT16
#endif
#endif

#if defined(LIBC_COMPILER_HAS_C23_FLOAT16)
using float16 = _Float16;
#define LIBC_HAS_FLOAT16
#endif

// float128 support.
#if (defined(LIBC_COMPILER_GCC_VER) && (LIBC_COMPILER_GCC_VER >= 1301)) &&     \
    (defined(LIBC_TARGET_ARCH_IS_AARCH64) ||                                   \
     defined(LIBC_TARGET_ARCH_IS_ANY_RISCV) ||                                 \
     defined(LIBC_TARGET_ARCH_IS_X86_64))
#define LIBC_COMPILER_HAS_C23_FLOAT128
#endif
#if (defined(LIBC_COMPILER_CLANG_VER) && (LIBC_COMPILER_CLANG_VER >= 500)) &&  \
    (defined(LIBC_TARGET_ARCH_IS_X86_64) &&                                    \
     !defined(LIBC_TARGET_OS_IS_FUCHSIA))
#define LIBC_COMPILER_HAS_FLOAT128_EXTENSION
#endif

#if defined(LIBC_COMPILER_HAS_C23_FLOAT128)
using float128 = _Float128;
#elif defined(LIBC_COMPILER_HAS_FLOAT128_EXTENSION)
using float128 = __float128;
#elif defined(LIBC_LONG_DOUBLE_IS_FLOAT128)
using float128 = long double;
#endif

#if defined(LIBC_COMPILER_HAS_C23_FLOAT128) ||                                 \
    defined(LIBC_COMPILER_HAS_FLOAT128_EXTENSION) ||                           \
    defined(LIBC_LONG_DOUBLE_IS_FLOAT128)
// TODO: Replace with LIBC_HAS_FLOAT128
#define LIBC_COMPILER_HAS_FLOAT128
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_FLOAT_H
