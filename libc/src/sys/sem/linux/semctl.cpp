//===-- Linux implementation of semctl ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/sem/semctl.h"

#include "hdr/errno_macros.h"
#include "hdr/sys_ipc_macros.h"
#include "hdr/sys_sem_macros.h"
#include "hdr/types/struct_semid_ds.h"
#include "hdr/types/struct_seminfo.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include <stdarg.h>
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, semctl, (int semid, int semnum, int cmd, ...)) {
  unsigned long cmd_arg = 0;

  switch (cmd) {
  case IPC_RMID:
  case GETVAL:
  case GETPID:
  case GETNCNT:
  case GETZCNT:
    break;

  case SETVAL: {
    va_list vargs;
    va_start(vargs, cmd);
    cmd_arg = static_cast<unsigned long>(va_arg(vargs, int));
    va_end(vargs);
    break;
  }

  case IPC_SET:
  case IPC_STAT:
  case SEM_STAT:
  case SEM_STAT_ANY: {
    va_list vargs;
    va_start(vargs, cmd);
    cmd_arg = reinterpret_cast<unsigned long>(va_arg(vargs, struct semid_ds *));
    va_end(vargs);
    break;
  }

  case GETALL:
  case SETALL: {
    va_list vargs;
    va_start(vargs, cmd);
    cmd_arg = reinterpret_cast<unsigned long>(va_arg(vargs, unsigned short *));
    va_end(vargs);
    break;
  }

  case IPC_INFO:
  case SEM_INFO: {
    va_list vargs;
    va_start(vargs, cmd);
    cmd_arg = reinterpret_cast<unsigned long>(va_arg(vargs, struct seminfo *));
    va_end(vargs);
    break;
  }

  default:
    libc_errno = EINVAL;
    return -1;
  }

  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_semctl, semid, semnum, cmd,
                                              cmd_arg);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
