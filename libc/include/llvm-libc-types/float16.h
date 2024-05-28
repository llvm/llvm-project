//===-- Definition of float16 type ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_FLOAT16_H
#define LLVM_LIBC_TYPES_FLOAT16_H

#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/compiler.h"
#include "src/__support/macros/properties/cpu_features.h"

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

#endif // LLVM_LIBC_TYPES_FLOAT16_H
