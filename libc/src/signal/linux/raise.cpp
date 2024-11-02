//===-- Linux implementation of signal ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/raise.h"
#include "src/signal/linux/signal_utils.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, raise, (int sig)) {
  ::sigset_t sigset;
  block_all_signals(sigset);
  long pid = __llvm_libc::syscall_impl(SYS_getpid);
  long tid = __llvm_libc::syscall_impl(SYS_gettid);
  int ret = __llvm_libc::syscall_impl(SYS_tgkill, pid, tid, sig);
  restore_signals(sigset);
  return ret;
}

} // namespace __llvm_libc
