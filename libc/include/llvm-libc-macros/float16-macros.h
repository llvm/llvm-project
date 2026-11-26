//===-- Detection of _Float16 compiler builtin type -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_FLOAT16_MACROS_H
#define LLVM_LIBC_MACROS_FLOAT16_MACROS_H

#include "../llvm-libc-types/float128.h"

// LIBC_TYPES_HAS_FLOAT16 is defined only when the compiler provides a usable
// native _Float16: GCC 13+ or Clang 12+, and not on targets where _Float16 is
// missing or has ABI problems (32-bit ARM, RISC-V, Windows, and i386 without
// SSE2). When it is not defined, the internal software type fputil::Float16
// is used instead.
#if defined(__FLT16_MANT_DIG__) &&                                             \
    (!defined(__GNUC__) || __GNUC__ >= 13 ||                                   \
     (defined(__clang__) && __clang_major__ >= 12)) &&                         \
    !defined(__arm__) && !defined(_M_ARM) && !defined(__riscv) &&              \
    !defined(_WIN32) && (!defined(__i386__) || defined(__SSE2__))
#define LIBC_TYPES_HAS_FLOAT16
#endif

// Defined when both native _Float16 and native float128 are available, for
// functions that take or return both types (e.g. f16sqrtf128).
// TODO: This would no longer be required if HdrGen let us guard function
// declarations with multiple macros.
#if defined(LIBC_TYPES_HAS_FLOAT16) && defined(LIBC_TYPES_HAS_NATIVE_FLOAT128)
#define LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128
#endif // LIBC_TYPES_HAS_FLOAT16 && LIBC_TYPES_HAS_NATIVE_FLOAT128

#endif // LLVM_LIBC_MACROS_FLOAT16_MACROS_H
