//===-- Definition of a common mutex type ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES___MUTEX_T_H
#define __LLVM_LIBC_TYPES___MUTEX_T_H

#include <llvm-libc-types/__futex_word.h>

typedef struct {
  unsigned char __timed;
  unsigned char __recursive;
  unsigned char __robust;

  void *__owner;
  unsigned long long __lock_count;

#ifdef __linux__
  __futex_word __ftxw;
#else
#error "Mutex type not defined for the target platform."
#endif
} __mutex_type;

#endif // __LLVM_LIBC_TYPES___MUTEX_T_H
