//===--- Futex Wrapper ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_FUTEX_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_FUTEX_UTILS_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/futex_word.h"
#include "src/__support/time/abs_timeout.h"
#include <linux/errno.h>
#include <linux/futex.h>

namespace LIBC_NAMESPACE_DECL {
class Futex : public cpp::Atomic<FutexWordType> {
public:
  using Timeout = internal::AbsTimeout;
  LIBC_INLINE constexpr Futex(FutexWordType value)
      : cpp::Atomic<FutexWordType>(value) {}
  LIBC_INLINE Futex &operator=(FutexWordType value) {
    cpp::Atomic<FutexWordType>::store(value);
    return *this;
  }
  LIBC_INLINE ErrorOr<int> wait(FutexWordType expected,
                                cpp::optional<Timeout> timeout = cpp::nullopt,
                                bool is_shared = false) {
    // use bitset variants to enforce abs_time
    uint32_t op = is_shared ? FUTEX_WAIT_BITSET : FUTEX_WAIT_BITSET_PRIVATE;
    if (timeout && timeout->is_realtime()) {
      op |= FUTEX_CLOCK_REALTIME;
    }
    for (;;) {
      if (this->load(cpp::MemoryOrder::RELAXED) != expected)
        return 0;

      int ret = syscall_impl<int>(
          /* syscall number */ FUTEX_SYSCALL_ID,
          /* futex address */ this,
          /* futex operation  */ op,
          /* expected value */ expected,
          /* timeout */ timeout ? &timeout->get_timespec() : nullptr,
          /* ignored */ nullptr,
          /* bitset */ FUTEX_BITSET_MATCH_ANY);

      // continue waiting if interrupted; otherwise return the result
      // which should normally be 0 or -ETIMEOUT. we also need to continue
      // if the error is EAGAIN, where the futex value may be detected as
      // changed spuriously, and we just poll again to ensure correctness.
      if (ret == -EINTR || ret == -EAGAIN)
        continue;

      if (ret < 0)
        return cpp::unexpected(-ret);
      return ret;
    }
  }
  LIBC_INLINE ErrorOr<int> notify_one(bool is_shared = false) {
    int ret = syscall_impl<int>(
        /* syscall number */ FUTEX_SYSCALL_ID,
        /* futex address */ this,
        /* futex operation  */ is_shared ? FUTEX_WAKE : FUTEX_WAKE_PRIVATE,
        /* wake up limit */ 1,
        /* ignored */ nullptr,
        /* ignored */ nullptr,
        /* ignored */ 0);
    if (ret < 0)
      return cpp::unexpected(-ret);
    return ret;
  }
  LIBC_INLINE ErrorOr<int> notify_all(bool is_shared = false) {
    int ret = syscall_impl<int>(
        /* syscall number */ FUTEX_SYSCALL_ID,
        /* futex address */ this,
        /* futex operation  */ is_shared ? FUTEX_WAKE : FUTEX_WAKE_PRIVATE,
        /* wake up limit */ cpp::numeric_limits<int>::max(),
        /* ignored */ nullptr,
        /* ignored */ nullptr,
        /* ignored */ 0);
    if (ret < 0)
      return cpp::unexpected(-ret);
    return ret;
  }
  LIBC_INLINE ErrorOr<int> requeue_to(Futex &other,
                                      cpp::optional<FutexWordType> oldval,
                                      int wake_limit, int requeue_limit,
                                      bool is_shared = false) {
    int ret;
    if (oldval)
      ret = syscall_impl<int>(
          /* syscall number */ FUTEX_SYSCALL_ID,
          /* futex address */ this,
          /* futex operation  */
          is_shared ? FUTEX_CMP_REQUEUE : FUTEX_CMP_REQUEUE_PRIVATE,
          /* wake up limit */ wake_limit,
          /* requeue limit */ requeue_limit,
          /* requeue address */ &other, *oldval);
    else
      ret = syscall_impl<int>(
          /* syscall number */ FUTEX_SYSCALL_ID,
          /* futex address */ this,
          /* futex operation  */
          is_shared ? FUTEX_REQUEUE : FUTEX_REQUEUE_PRIVATE,
          /* wake up limit */ wake_limit,
          /* requeue limit */
          requeue_limit,
          /* requeue address */
          &other);
    if (ret < 0)
      return cpp::unexpected<int>(-ret);
    return static_cast<int>(ret);
  }
};

static_assert(__is_standard_layout(Futex),
              "Futex must be a standard layout type.");
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_FUTEX_UTILS_H
