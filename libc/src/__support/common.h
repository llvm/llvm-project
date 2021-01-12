//===-- Common internal contructs -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SUPPORT_COMMON_H
#define LLVM_LIBC_SUPPORT_COMMON_H

#define LIBC_INLINE_ASM __asm__ __volatile__

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(x, 0)
#define UNUSED __attribute__((unused))

#ifndef LLVM_LIBC_FUNCTION_ATTR
#define LLVM_LIBC_FUNCTION_ATTR
#endif

#ifdef LLVM_LIBC_PUBLIC_PACKAGING
#define LLVM_LIBC_FUNCTION(type, name, arglist)                                \
  LLVM_LIBC_FUNCTION_ATTR decltype(__llvm_libc::name)                          \
      __##name##_impl__ __asm__(#name);                                        \
  decltype(__llvm_libc::name) name [[gnu::alias(#name)]];                      \
  type __##name##_impl__ arglist
#else
#define LLVM_LIBC_FUNCTION(type, name, arglist) type name arglist
#endif

#endif // LLVM_LIBC_SUPPORT_COMMON_H
