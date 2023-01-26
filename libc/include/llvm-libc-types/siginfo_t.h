//===-- Definition of siginfo_t type --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_SIGINFO_T_H__
#define __LLVM_LIBC_TYPES_SIGINFO_T_H__

#include <llvm-libc-types/pid_t.h>
#include <llvm-libc-types/uid_t.h>
#include <llvm-libc-types/union_sigval.h>

typedef struct {
  int si_signo;
  int si_code;
  int si_errno;
  pid_t si_pid;
  uid_t si_uid;
  void *si_addr;
  int si_status;
  long si_band;
  union sigval si_value;
} siginfo_t;

#endif // __LLVM_LIBC_TYPES_SIGINFO_T_H__
