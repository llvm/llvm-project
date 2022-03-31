//===-- Linux implementation of sigaction ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __LLVM_LIBC_INTERNAL_SIGACTION
#include "src/signal/sigaction.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/signal/linux/signal.h"

#include "src/__support/common.h"

namespace __llvm_libc {

// TOOD: Some architectures will have their signal trampoline functions in the
// vdso, use those when available.

extern "C" void __restore_rt();

template <typename T, typename V>
static void copy_sigaction(T &dest, const V &source) {
  dest.sa_handler = source.sa_handler;
  dest.sa_mask = source.sa_mask;
  dest.sa_flags = source.sa_flags;
  dest.sa_restorer = source.sa_restorer;
}

LLVM_LIBC_FUNCTION(int, sigaction,
                   (int signal, const struct __sigaction *__restrict libc_new,
                    struct __sigaction *__restrict libc_old)) {
  struct sigaction kernel_new;
  if (libc_new) {
    copy_sigaction(kernel_new, *libc_new);
    if (!(kernel_new.sa_flags & SA_RESTORER)) {
      kernel_new.sa_flags |= SA_RESTORER;
      kernel_new.sa_restorer = __restore_rt;
    }
  }

  struct sigaction kernel_old;
  int ret = syscall(SYS_rt_sigaction, signal, libc_new ? &kernel_new : nullptr,
                    libc_old ? &kernel_old : nullptr, sizeof(sigset_t));
  if (ret) {
    llvmlibc_errno = -ret;
    return -1;
  }

  if (libc_old)
    copy_sigaction(*libc_old, kernel_old);
  return 0;
}

} // namespace __llvm_libc
