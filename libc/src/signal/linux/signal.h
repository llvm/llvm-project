//===-- Internal header for Linux signals -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_LINUX_SIGNAL_H
#define LLVM_LIBC_SRC_SIGNAL_LINUX_SIGNAL_H

#include "config/linux/syscall.h" // For internal syscall function.
#include "include/sys/syscall.h"  // For syscall numbers.

#include "include/signal.h"

static_assert(sizeof(sigset_t) * 8 >= NSIG, "sigset_t cannot hold all signals");

namespace __llvm_libc {

// Using this internally defined type will make it easier in the future to port
// to different architectures.
struct Sigset {
  sigset_t nativeSigset;

  constexpr static Sigset fullset() { return {-1UL}; }
  constexpr static Sigset emptySet() { return {0}; }

  constexpr void addset(int signal) { nativeSigset |= (1L << (signal - 1)); }

  constexpr void delset(int signal) { nativeSigset &= ~(1L << (signal - 1)); }

  operator sigset_t() const { return nativeSigset; }
};

constexpr static Sigset all = Sigset::fullset();

static inline int block_all_signals(Sigset &set) {
  sigset_t nativeSigset = all;
  sigset_t oldSet = set;
  int ret = __llvm_libc::syscall(SYS_rt_sigprocmask, SIG_BLOCK, &nativeSigset,
                                 &oldSet, sizeof(sigset_t));
  set = {oldSet};
  return ret;
}

static inline int restore_signals(const Sigset &set) {
  sigset_t nativeSigset = set;
  return __llvm_libc::syscall(SYS_rt_sigprocmask, SIG_SETMASK, &nativeSigset,
                              nullptr, sizeof(sigset_t));
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SIGNAL_LINUX_SIGNAL_H
