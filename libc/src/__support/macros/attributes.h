//===-- Portable attributes -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SUPPORT_MACROS_ATTRIBUTES_H
#define LLVM_LIBC_SUPPORT_MACROS_ATTRIBUTES_H

#define LIBC_INLINE inline
#define LIBC_INLINE_ASM __asm__ __volatile__
#define LIBC_LIKELY(x) __builtin_expect(!!(x), 1)
#define LIBC_UNLIKELY(x) __builtin_expect(x, 0)
#define LIBC_UNUSED __attribute__((unused))

#endif // LLVM_LIBC_SUPPORT_MACROS_ATTRIBUTES_H
