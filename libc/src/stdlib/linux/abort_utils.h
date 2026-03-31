//===-- Internal header for Linux abort -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_LINUX_ABORT_UTILS_H
#define LLVM_LIBC_SRC_STDLIB_LINUX_ABORT_UTILS_H

#include "hdr/types/sigset_t.h"
#include "include/llvm-libc-types/sigset_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/raise.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/rwlock.h"
#include "src/signal/linux/signal_utils.h"

namespace LIBC_NAMESPACE_DECL {

namespace abort_utils {

// TODO: this lock needs to be acquired during _Fork/fork.
class AbortLockGuard {
private:
  sigset_t old_mask;
  LIBC_INLINE_VAR static RwLock abort_lock;

public:
  LIBC_INLINE constexpr AbortLockGuard(bool exclusive) : old_mask{} {
    RwLock::LockResult result = RwLock::LockResult::Success;
    do {
      if (exclusive)
        result = abort_lock.write_lock(cpp::nullopt);
      else
        result = abort_lock.read_lock(cpp::nullopt);
    } while (result == RwLock::LockResult::Overflow);

    (void)block_all_signals(old_mask);
  }

  LIBC_INLINE ~AbortLockGuard() {
    (void)restore_signals(old_mask);
    (void)abort_lock.unlock();
  }
};

[[noreturn]] LIBC_INLINE void abort() {
  // 1. Try to raise SIGABRT.
  (void)LIBC_NAMESPACE::linux_syscalls::raise(SIGABRT);

  // We get back from abort, potentially from a abort handler.
  // We recover the handler to default and raise it again. Since this is the
  // real abort routine, we demand exclusive access to the abort lock.
  // We have already returned from the first raise, so it is okay to grab
  // exclusive access.
  AbortLockGuard guard(true);
  struct sigaction sa{};
  sa.sa_handler = SIG_DFL;
  sa.sa_flags = 0;
  (void)do_sigaction(SIGABRT, &sa, nullptr);
  (void)LIBC_NAMESPACE::linux_syscalls::raise(SIGABRT);

  // Now unblock the signal. The pending abort signal is now unblocked and
  // should be delivered to its default handler.
  (void)unblock_signal(SIGABRT);

  LIBC_NAMESPACE::internal::exit(127);
}
} // namespace abort_utils

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_LINUX_ABORT_UTILS_H
