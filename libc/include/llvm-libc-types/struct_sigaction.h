//===-- Definition of struct __sigaction ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_SIGACTION_H__
#define __LLVM_LIBC_TYPES_SIGACTION_H__

struct __sigaction {
  union {
    void (*sa_handler)(int);
    void (*sa_action)(int, siginfo_t *, void *);
  };
  sigset_t sa_mask;
  int sa_flags;
  void (*sa_restorer)(void);
};

#endif // __LLVM_LIBC_TYPES_SIGACTION_H__
