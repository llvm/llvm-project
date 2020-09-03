//===-- Linux implementation of sigprocmask -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigprocmask.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/signal/linux/signal.h"

#include "src/__support/common.h"

namespace __llvm_libc {

int LLVM_LIBC_ENTRYPOINT(sigprocmask)(int how, const sigset_t *__restrict set,
                                      sigset_t *__restrict oldset) {
  int ret = __llvm_libc::syscall(SYS_rt_sigprocmask, how, set, oldset,
                                 sizeof(sigset_t));
  if (!ret)
    return 0;

  llvmlibc_errno = -ret;
  return -1;
}

} // namespace __llvm_libc
