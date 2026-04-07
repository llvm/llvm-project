//===-- Definition of sem_t type -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_SEM_T_H
#define LLVM_LIBC_TYPES_SEM_T_H

#include "__futex_word.h"

typedef struct {
  __futex_word __value;        // current semaphore count
  unsigned int __canary;       // used for sanity check
  unsigned char __reserved[8]; // for future usage, total fixed size 16 bytes
} sem_t;

#endif // LLVM_LIBC_TYPES_SEM_T_H
