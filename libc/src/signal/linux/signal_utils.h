//===-- Internal header for Linux signals -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_LINUX_SIGNAL_UTILS_H
#define LLVM_LIBC_SRC_SIGNAL_LINUX_SIGNAL_UTILS_H

#include "hdr/signal_macros.h"
#include "hdr/types/siginfo_t.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_sigaction.h"
#include "src/__support/OSUtil/linux/vdso.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/raw_rwlock.h"

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

extern "C" void __restore_rt();

// The POSIX definition of struct sigaction and the sigaction data structure
// expected by the rt_sigaction syscall differ in their definition. So, we
// define the equivalent of the what the kernel expects to help with making
// the rt_sigaction syscall.
//
// NOTE: Though the kernel definition does not have a union to include the
// handler taking siginfo_t * argument, one can set sa_handler to sa_sigaction
// if SA_SIGINFO is set in sa_flags.
struct KernelSigaction {
  LIBC_INLINE KernelSigaction &operator=(const struct sigaction &sa) {
    sa_flags = sa.sa_flags;
    sa_restorer = sa.sa_restorer;
    sa_mask = sa.sa_mask;
    if (sa_flags & SA_SIGINFO) {
      sa_sigaction = sa.sa_sigaction;
    } else {
      sa_handler = sa.sa_handler;
    }
    return *this;
  }

  LIBC_INLINE operator struct sigaction() const {
    struct sigaction sa;
    sa.sa_flags = static_cast<int>(sa_flags);
    sa.sa_mask = sa_mask;
    sa.sa_restorer = sa_restorer;
    if (sa_flags & SA_SIGINFO)
      sa.sa_sigaction = sa_sigaction;
    else
      sa.sa_handler = sa_handler;
    return sa;
  }

  union {
    void (*sa_handler)(int);
    void (*sa_sigaction)(int, siginfo_t *, void *);
  };
  unsigned long sa_flags;
  void (*sa_restorer)(void);
  // Our public definition of sigset_t matches that of the kernel's definition.
  // So, we can use the public sigset_t type here.
  sigset_t sa_mask;
};

static constexpr size_t BITS_PER_SIGWORD = sizeof(unsigned long) * 8;

LIBC_INLINE constexpr sigset_t full_set() { return sigset_t{{-1UL}}; }

LIBC_INLINE constexpr sigset_t empty_set() { return sigset_t{{0}}; }

// Set the bit corresponding to |signal| in |set|. Return true on success
// and false on failure. The function will fail if |signal| is greater than
// NSIG or negative.
LIBC_INLINE constexpr bool add_signal(sigset_t &set, int signal) {
  if (signal > NSIG || signal <= 0)
    return false;
  size_t n = size_t(signal) - 1;
  size_t word = n / BITS_PER_SIGWORD;
  size_t bit = n % BITS_PER_SIGWORD;
  set.__signals[word] |= (1UL << bit);
  return true;
}

// Reset the bit corresponding to |signal| in |set|. Return true on success
// and false on failure. The function will fail if |signal| is greater than
// NSIG or negative.
LIBC_INLINE constexpr bool delete_signal(sigset_t &set, int signal) {
  if (signal > NSIG || signal <= 0)
    return false;
  size_t n = size_t(signal) - 1;
  size_t word = n / BITS_PER_SIGWORD;
  size_t bit = n % BITS_PER_SIGWORD;
  set.__signals[word] &= ~(1UL << bit);
  return true;
}

LIBC_INLINE int block_all_signals(sigset_t &set) {
  sigset_t full = full_set();
  return LIBC_NAMESPACE::syscall_impl<int>(SYS_rt_sigprocmask, SIG_BLOCK, &full,
                                           &set, sizeof(sigset_t));
}

LIBC_INLINE int restore_signals(const sigset_t &set) {
  return LIBC_NAMESPACE::syscall_impl<int>(SYS_rt_sigprocmask, SIG_SETMASK,
                                           &set, nullptr, sizeof(sigset_t));
}

LIBC_INLINE int unblock_signal(int signal) {
  sigset_t set = empty_set();
  if (!add_signal(set, signal))
    return -EINVAL;
  return LIBC_NAMESPACE::syscall_impl<int>(SYS_rt_sigprocmask, SIG_UNBLOCK,
                                           &set, nullptr, sizeof(sigset_t));
}

// This guard is used to:
// 1. temporarily block the all signal, avoid post fork invalid state to be
//    exposed to async signal handlers.
// 2. ensure the ordering between sigaction and fork/spawn, so that forked
//    processes can see modification from a just returned concurrent call.
class SigAbortGuard {
private:
  sigset_t old_mask;
  LIBC_INLINE_VAR static RawRwLock abort_lock;

public:
  LIBC_INLINE constexpr SigAbortGuard(bool exclusive) : old_mask{} {
    RawRwLock::LockResult result = RawRwLock::LockResult::Success;
    do {
      if (exclusive)
        result = abort_lock.write_lock(cpp::nullopt);
      else
        result = abort_lock.read_lock(cpp::nullopt);
    } while (result == RawRwLock::LockResult::Overflow);

    // This uses a valid sigset_t size and internal storage. A failure here
    // would indicate a kernel ABI mismatch, which is not actionable here.
    block_all_signals(old_mask);
  }

  LIBC_INLINE ~SigAbortGuard() {
    // This restores a previously saved mask from internal storage. A failure
    // here would likewise be a non-recoverable kernel ABI issue.
    restore_signals(old_mask);
    (void)abort_lock.unlock();
  }
};

LIBC_INLINE ErrorOr<int>
unchecked_sigaction(int signal, const struct sigaction *__restrict libc_new,
                    struct sigaction *__restrict libc_old) {
  vdso::TypedSymbol<vdso::VDSOSym::RTSigReturn> rt_sigreturn;
  KernelSigaction kernel_new;
  if (libc_new) {
    kernel_new = *libc_new;
    if (!(kernel_new.sa_flags & SA_RESTORER)) {
      kernel_new.sa_flags |= SA_RESTORER;
      kernel_new.sa_restorer = rt_sigreturn ? rt_sigreturn : __restore_rt;
    }
  }

  KernelSigaction kernel_old;
  int ret = LIBC_NAMESPACE::syscall_impl<int>(
      SYS_rt_sigaction, signal, libc_new ? &kernel_new : nullptr,
      libc_old ? &kernel_old : nullptr, sizeof(sigset_t));
  if (ret)
    return Error(-ret);

  if (libc_old)
    *libc_old = kernel_old;
  return 0;
}

LIBC_INLINE ErrorOr<int>
checked_sigaction(int signal, const struct sigaction *__restrict libc_new,
                  struct sigaction *__restrict libc_old) {
  if (signal <= 0 || signal >= NSIG)
    return Error(EINVAL);
  if (signal == SIGABRT) {
    SigAbortGuard guard(true);
    return unchecked_sigaction(signal, libc_new, libc_old);
  }
  return unchecked_sigaction(signal, libc_new, libc_old);
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SIGNAL_LINUX_SIGNAL_UTILS_H
