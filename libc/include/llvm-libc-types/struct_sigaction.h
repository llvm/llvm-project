//===-- Definition of struct __sigaction ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_SIGACTION_H__
#define __LLVM_LIBC_TYPES_SIGACTION_H__

#include <llvm-libc-types/siginfo_t.h>
#include <llvm-libc-types/sigset_t.h>

struct sigaction {
  union {
    void (*sa_handler)(int);
    void (*sa_sigaction)(int, siginfo_t *, void *);
  };
  sigset_t sa_mask;
  int sa_flags;
#ifdef __linux__
  // This field is present on linux for most targets.
  void (*sa_restorer)(void);
#endif
};

typedef void (*__sighandler_t)(int);

#endif // __LLVM_LIBC_TYPES_SIGACTION_H__
