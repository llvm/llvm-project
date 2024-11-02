//===-- Definition of off_t type ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_OFF_T_H__
#define __LLVM_LIBC_TYPES_OFF_T_H__

#if defined(__LP64__) || defined(__riscv)
typedef __INT64_TYPE__ off_t;
#else
typedef __INT32_TYPE__ off_t;
#endif // __LP64__ || __riscv

#endif // __LLVM_LIBC_TYPES_OFF_T_H__
