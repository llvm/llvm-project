//===-- Definition of thrd_t type -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_THRD_T_H__
#define __LLVM_LIBC_TYPES_THRD_T_H__

#include <llvm-libc-types/__futex_word.h>

typedef struct {
  __futex_word __clear_tid;
  int __tid;
  void *__stack;
  int __stack_size;
  int __retval;
} thrd_t;

#endif // __LLVM_LIBC_TYPES_THRD_T_H__
