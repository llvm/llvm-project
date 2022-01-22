//===-- Definition of type fenv_t -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_FENV_T_H__
#define __LLVM_LIBC_TYPES_FENV_T_H__

#ifdef __aarch64__
typedef struct {
  unsigned char __control_word[4];
  unsigned char __status_word[4];
} fenv_t;
#endif
#ifdef __x86_64__
typedef struct {
  unsigned char __x86_status[28];
  unsigned char __mxcsr[4];
} fenv_t;
#endif

#endif // __LLVM_LIBC_TYPES_FENV_T_H__
