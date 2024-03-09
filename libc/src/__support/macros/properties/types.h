//===-- Types support -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Types detection and support.

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_TYPES_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_TYPES_H

#include "include/llvm-libc-macros/float-macros.h" // LDBL_MANT_DIG
#include "include/llvm-libc-types/float128.h"      // float128
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/compiler.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/__support/macros/properties/os.h"

#include <stdint.h> // UINT64_MAX, __SIZEOF_INT128__

// 'long double' properties.
#if (LDBL_MANT_DIG == 53)
#define LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64
#elif (LDBL_MANT_DIG == 64)
#define LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
#elif (LDBL_MANT_DIG == 113)
#define LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128
#endif

// int64 / uint64 support
#if defined(UINT64_MAX)
#define LIBC_TYPES_HAS_INT64
#endif // UINT64_MAX

// int128 / uint128 support
#if defined(__SIZEOF_INT128__)
#define LIBC_TYPES_HAS_INT128
#endif // defined(__SIZEOF_INT128__)

// -- float16 support ---------------------------------------------------------
// TODO: move this logic to "llvm-libc-types/float16.h"
#if defined(LIBC_TARGET_ARCH_IS_X86_64) && defined(LIBC_TARGET_CPU_HAS_SSE2)
#if (defined(LIBC_COMPILER_CLANG_VER) && (LIBC_COMPILER_CLANG_VER >= 1500)) || \
    (defined(LIBC_COMPILER_GCC_VER) && (LIBC_COMPILER_GCC_VER >= 1201))
#define LIBC_TYPES_HAS_FLOAT16
using float16 = _Float16;
#endif
#endif
#if defined(LIBC_TARGET_ARCH_IS_AARCH64)
#if (defined(LIBC_COMPILER_CLANG_VER) && (LIBC_COMPILER_CLANG_VER >= 900)) ||  \
    (defined(LIBC_COMPILER_GCC_VER) && (LIBC_COMPILER_GCC_VER >= 1301))
#define LIBC_TYPES_HAS_FLOAT16
using float16 = _Float16;
#endif
#endif
#if defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#if (defined(LIBC_COMPILER_CLANG_VER) && (LIBC_COMPILER_CLANG_VER >= 1300)) || \
    (defined(LIBC_COMPILER_GCC_VER) && (LIBC_COMPILER_GCC_VER >= 1301))
#define LIBC_TYPES_HAS_FLOAT16
using float16 = _Float16;
#endif
#endif

// -- float128 support --------------------------------------------------------
// LIBC_TYPES_HAS_FLOAT128 and 'float128' type are provided by
// "include/llvm-libc-types/float128.h"

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_TYPES_H
