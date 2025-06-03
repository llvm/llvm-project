//===-- Detection of __bf16 compiler builtin type -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_BFLOAT16_MACROS_H
#define LLVM_LIBC_MACROS_BFLOAT16_MACROS_H

#if ((defined(__clang__) && __clang_major__ > 17) ||                           \
     (defined(__GNUC__) && __GNUC__ > 13)) &&                                  \
    !defined(__arm__) && !defined(_M_ARM) && !defined(__riscv) &&              \
    !defined(_WIN32)

#define LIBC_TYPES_HAS_BFLOAT16

#endif // LIBC_TYPES_HAS_BFLOAT16

#endif // LLVM_LIBC_MACROS_BFLOAT16_MACROS_H
