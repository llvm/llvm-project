//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of system for Linux.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/system.h"
#include "hdr/signal_macros.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/linux/shell_fork.h"
#include "src/__support/OSUtil/linux/sigmask.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/rt_sigaction.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/signal/linux/signal_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, system, (const char *command)) {
  if (command == nullptr)
    return 1;

  KernelSigaction sa_ign{};
  sa_ign.sa_handler = SIG_IGN;

  KernelSigaction orig_int{};
  KernelSigaction orig_quit{};

  if (auto res = linux_syscalls::rt_sigaction(SIGINT, &sa_ign, &orig_int);
      !res.has_value()) {
    libc_errno = res.error();
    return -1;
  }

  if (auto res = linux_syscalls::rt_sigaction(SIGQUIT, &sa_ign, &orig_quit);
      !res.has_value()) {
    linux_syscalls::rt_sigaction(SIGINT, &orig_int, nullptr);
    libc_errno = res.error();
    return -1;
  }

  sigset_t orig_mask{};
  if (auto res = signals::block(SIGCHLD, &orig_mask); !res.has_value()) {
    linux_syscalls::rt_sigaction(SIGINT, &orig_int, nullptr);
    linux_syscalls::rt_sigaction(SIGQUIT, &orig_quit, nullptr);
    libc_errno = res.error();
    return -1;
  }

  auto restore_signals = [&] {
    linux_syscalls::rt_sigaction(SIGINT, &orig_int, nullptr);
    linux_syscalls::rt_sigaction(SIGQUIT, &orig_quit, nullptr);
    signals::set_mask(orig_mask);
  };

  // The child restores the original signal state before running the shell.
  ErrorOr<pid_t> fork_res = shell_fork(command, [&] {
    restore_signals();
    return true;
  });
  if (!fork_res.has_value()) {
    restore_signals();
    libc_errno = fork_res.error();
    return -1;
  }

  ErrorOr<int> wait_res = shell_wait(fork_res.value());

  restore_signals();

  if (!wait_res.has_value()) {
    libc_errno = wait_res.error();
    return -1;
  }
  return wait_res.value();
}

} // namespace LIBC_NAMESPACE_DECL
