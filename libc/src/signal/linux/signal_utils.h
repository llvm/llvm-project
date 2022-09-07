//===-- Internal header for Linux signals -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_LINUX_SIGNAL_UTILS_H
#define LLVM_LIBC_SRC_SIGNAL_LINUX_SIGNAL_UTILS_H

#include "include/sys/syscall.h"          // For syscall numbers.
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.

#include <signal.h>
#include <stddef.h>

namespace __llvm_libc {

// The POSIX definition of struct sigaction and the sigaction data structure
// expected by the rt_sigaction syscall differ in their definition. So, we
// define the equivalent of the what the kernel expects to help with making
// the rt_sigaction syscall.
//
// NOTE: Though the kernel definition does not have a union to include the
// handler taking siginfo_t * argument, one can set sa_handler to sa_sigaction
// if SA_SIGINFO is set in sa_flags.
struct KernelSigaction {
  using HandlerType = void(int);
  using SiginfoHandlerType = void(int, siginfo_t *, void *);

  KernelSigaction &operator=(const struct sigaction &sa) {
    sa_flags = sa.sa_flags;
    sa_restorer = sa.sa_restorer;
    sa_mask = sa.sa_mask;
    if (sa_flags & SA_SIGINFO) {
      sa_handler = reinterpret_cast<HandlerType *>(sa.sa_sigaction);
    } else {
      sa_handler = sa.sa_handler;
    }
    return *this;
  }

  operator struct sigaction() const {
    struct sigaction sa;
    sa.sa_flags = sa_flags;
    sa.sa_mask = sa_mask;
    sa.sa_restorer = sa_restorer;
    if (sa_flags & SA_SIGINFO)
      sa.sa_sigaction = reinterpret_cast<SiginfoHandlerType *>(sa_handler);
    else
      sa.sa_handler = sa_handler;
    return sa;
  }

  HandlerType *sa_handler;
  unsigned long sa_flags;
  void (*sa_restorer)(void);
  // Our public definition of sigset_t matches that of the kernel's definition.
  // So, we can use the public sigset_t type here.
  sigset_t sa_mask;
};

static constexpr size_t BITS_PER_SIGWORD = sizeof(unsigned long) * 8;

constexpr sigset_t full_set() { return sigset_t{{-1UL}}; }

constexpr sigset_t empty_set() { return sigset_t{{0}}; }

// Set the bit corresponding to |signal| in |set|. Return true on success
// and false on failure. The function will fail if |signal| is greater than
// NSIG or negative.
constexpr inline bool add_signal(sigset_t &set, int signal) {
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
constexpr inline bool delete_signal(sigset_t &set, int signal) {
  if (signal > NSIG || signal <= 0)
    return false;
  size_t n = size_t(signal) - 1;
  size_t word = n / BITS_PER_SIGWORD;
  size_t bit = n % BITS_PER_SIGWORD;
  set.__signals[word] &= ~(1UL << bit);
  return true;
}

static inline int block_all_signals(sigset_t &set) {
  sigset_t full = full_set();
  return __llvm_libc::syscall(SYS_rt_sigprocmask, SIG_BLOCK, &full, &set,
                              sizeof(sigset_t));
}

static inline int restore_signals(const sigset_t &set) {
  return __llvm_libc::syscall(SYS_rt_sigprocmask, SIG_SETMASK, &set, nullptr,
                              sizeof(sigset_t));
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SIGNAL_LINUX_SIGNAL_UTILS_H
