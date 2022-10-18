//===-- Definition of stack_t type ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_STACK_T_H__
#define __LLVM_LIBC_TYPES_STACK_T_H__

#include <llvm-libc-types/size_t.h>

typedef struct {
  // The order of the fields declared here should match the kernel definition
  // of stack_t in order for the SYS_sigaltstack syscall to work correctly.
  void *ss_sp;
  int ss_flags;
  size_t ss_size;
} stack_t;

#endif // __LLVM_LIBC_TYPES_STACK_T_H__
