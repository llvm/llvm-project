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
#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/execle.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/fork.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/rt_sigaction.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/rt_sigprocmask.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/wait4.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/signal/linux/signal_utils.h"
#include "src/unistd/environ.h"

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

  sigset_t block_mask{};
  add_signal(block_mask, SIGCHLD);
  sigset_t orig_mask{};
  if (auto res =
          linux_syscalls::rt_sigprocmask(SIG_BLOCK, &block_mask, &orig_mask);
      !res.has_value()) {
    linux_syscalls::rt_sigaction(SIGINT, &orig_int, nullptr);
    linux_syscalls::rt_sigaction(SIGQUIT, &orig_quit, nullptr);
    libc_errno = res.error();
    return -1;
  }

  auto fork_res = linux_syscalls::fork();
  if (!fork_res.has_value()) {
    linux_syscalls::rt_sigaction(SIGINT, &orig_int, nullptr);
    linux_syscalls::rt_sigaction(SIGQUIT, &orig_quit, nullptr);
    linux_syscalls::rt_sigprocmask(SIG_SETMASK, &orig_mask, nullptr);
    libc_errno = fork_res.error();
    return -1;
  }

  pid_t pid = fork_res.value();
  if (pid == 0) {
    linux_syscalls::rt_sigaction(SIGINT, &orig_int, nullptr);
    linux_syscalls::rt_sigaction(SIGQUIT, &orig_quit, nullptr);
    linux_syscalls::rt_sigprocmask(SIG_SETMASK, &orig_mask, nullptr);

    // Error checking isn't helpful since this is the forked process, so we
    // can't set errno. All we can meaningfully do is exit with status 127.
    linux_syscalls::execle("/bin/sh", "sh", "-c", command, nullptr, environ);

    internal::exit(127);
  }

  int status = 0;
  int wait_ret = 0;
  do {
    if (auto wait_res = linux_syscalls::wait4(pid, &status, 0, nullptr);
        !wait_res.has_value()) {
      if (wait_res.error() == EINTR)
        continue;
      wait_ret = -1;
      libc_errno = wait_res.error();
      break;
    }
    wait_ret = status;
    break;
  } while (true);

  linux_syscalls::rt_sigaction(SIGINT, &orig_int, nullptr);
  linux_syscalls::rt_sigaction(SIGQUIT, &orig_quit, nullptr);
  linux_syscalls::rt_sigprocmask(SIG_SETMASK, &orig_mask, nullptr);

  return wait_ret;
}

} // namespace LIBC_NAMESPACE_DECL
