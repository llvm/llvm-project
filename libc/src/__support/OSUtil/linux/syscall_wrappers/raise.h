//===-- Implementation header for raise -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_RAISE_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_RAISE_H

#include "hdr/signal_macros.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_impl
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> raise(int sig) {
  sigset_t full_set = sigset_t{{-1UL}};
#ifdef LIBC_COMPILER_IS_CLANG
  [[clang::uninitialized]] sigset_t old_set;
#else
  sigset_t old_set;
#endif
  auto restore_signals = [&old_set]() {
    return syscall_impl<int>(SYS_rt_sigprocmask, SIG_SETMASK, &old_set, nullptr,
                             sizeof(sigset_t));
  };

  int mask_result = syscall_impl<int>(SYS_rt_sigprocmask, SIG_BLOCK, &full_set,
                                      &old_set, sizeof(sigset_t));
  if (mask_result < 0)
    return Error(-mask_result);

  long pid = syscall_impl<long>(SYS_getpid);
  if (pid < 0) {
    restore_signals();
    return Error(-static_cast<int>(pid));
  }

  long tid = syscall_impl<long>(SYS_gettid);
  if (tid < 0) {
    restore_signals();
    return Error(-static_cast<int>(tid));
  }

  int result = syscall_impl<int>(SYS_tgkill, pid, tid, sig);
  int restore_result = restore_signals();
  if (result < 0)
    return Error(-result);
  if (restore_result < 0)
    return Error(-restore_result);
  return result;
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_RAISE_H
