//===--- Definition of a type for a futex word ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_FUTEX_WORD_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_FUTEX_WORD_H

#include "include/llvm-libc-types/struct_timespec.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/errno/libc_errno.h"
#include <linux/futex.h>
#include <stdint.h>
#include <sys/syscall.h>
namespace LIBC_NAMESPACE {

// Futexes are 32 bits in size on all platforms, including 64-bit platforms.
using FutexWordType = uint32_t;

#if SYS_futex
constexpr auto FUTEX_SYSCALL_ID = SYS_futex;
#elif defined(SYS_futex_time64)
constexpr auto FUTEX_SYSCALL_ID = SYS_futex_time64;
#else
#error "futex and futex_time64 syscalls not available."
#endif

// Returns false on timeout, and true in all other cases.
LIBC_INLINE bool futex_wait(cpp::Atomic<FutexWordType> &futex,
                            FutexWordType expected,
                            cpp::optional<::timespec> abs_timeout,
                            bool is_shared = false) {
  for (;;) {
    if (futex.load(cpp::MemoryOrder::RELAXED) != expected)
      return true;
    // Use FUTEX_WAIT_BITSET rather than FUTEX_WAIT to be able to give an
    // absolute time rather than a relative time.
    long ret =
        syscall_impl<long>(FUTEX_SYSCALL_ID, &futex,
                           is_shared ? FUTEX_WAIT_BITSET
                                     : (FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG),
                           expected, abs_timeout ? &*abs_timeout : nullptr,
                           nullptr, FUTEX_BITSET_MATCH_ANY);
    switch (ret) {
    case -EINTR:
      continue;
    case -ETIMEDOUT:
      return false;
    default:
      return true;
    }
  }
}

LIBC_INLINE bool futex_wake_one(cpp::Atomic<FutexWordType> &futex,
                                bool is_shared = false) {
  long ret = syscall_impl<long>(FUTEX_SYSCALL_ID, &futex,
                                is_shared ? FUTEX_WAKE : FUTEX_WAKE_PRIVATE, 1,
                                0, 0, 0);
  return ret > 0;
}

LIBC_INLINE void futex_wake_all(cpp::Atomic<FutexWordType> &futex,
                                bool is_shared = false) {
  syscall_impl<long>(FUTEX_SYSCALL_ID, &futex,
                     is_shared ? FUTEX_WAKE : FUTEX_WAKE_PRIVATE,
                     cpp::numeric_limits<int>::max(), 0, 0, 0);
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_FUTEX_WORD_H
