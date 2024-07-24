//===-- Linux implementation of sigaction ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigaction.h"

#include "hdr/types/sigset_t.h"
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"
#include "src/signal/linux/signal_utils.h"

namespace LIBC_NAMESPACE {

// TOOD: Some architectures will have their signal trampoline functions in the
// vdso, use those when available.

extern "C" void __restore_rt();

LLVM_LIBC_FUNCTION(int, sigaction,
                   (int signal, const struct sigaction *__restrict libc_new,
                    struct sigaction *__restrict libc_old)) {
  KernelSigaction kernel_new;
  if (libc_new) {
    kernel_new = *libc_new;
    if (!(kernel_new.sa_flags & SA_RESTORER)) {
      kernel_new.sa_flags |= SA_RESTORER;
      kernel_new.sa_restorer = __restore_rt;
    }
  }

  KernelSigaction kernel_old;
  int ret = LIBC_NAMESPACE::syscall_impl<int>(
      SYS_rt_sigaction, signal, libc_new ? &kernel_new : nullptr,
      libc_old ? &kernel_old : nullptr, sizeof(sigset_t));
  if (ret) {
    libc_errno = -ret;
    return -1;
  }

  if (libc_old)
    *libc_old = kernel_old;
  return 0;
}

} // namespace LIBC_NAMESPACE
