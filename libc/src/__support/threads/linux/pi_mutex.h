//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux priority inheritance mutex support.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_PI_MUTEX_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_PI_MUTEX_H

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/futex.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/threads/identifier.h"
#include "src/__support/threads/linux/futex_utils.h"
#include "src/__support/threads/linux/futex_word.h"
#include "src/__support/threads/mutex_common.h"

#include <linux/futex.h>

#ifdef LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
#include "src/__support/time/monotonicity.h"
#endif

namespace LIBC_NAMESPACE_DECL {

class PIMutexRef {
public:
  enum class Type { Normal, ErrorChecking, Recursive };

protected:
  Futex &owner;
  Type type;
  // Number of recursive locks minus one. This pointer must be provided
  // if type == Recursive
  size_t *recursive_count;

public:
  LIBC_INLINE PIMutexRef(Futex &owner, Type type,
                         size_t *recursive_count = nullptr)
      : owner(owner), type(type), recursive_count(recursive_count) {
    if (type == Type::Recursive)
      LIBC_CRASH_ON_NULLPTR(recursive_count);
  }
  LIBC_INLINE MutexError try_lock() {
    FutexWordType old_owner = 0;
    auto current = static_cast<FutexWordType>(internal::gettid());
    if (owner.compare_exchange_strong(old_owner, current,
                                      cpp::MemoryOrder::ACQUIRE,
                                      cpp::MemoryOrder::RELAXED))
      return MutexError::NONE;

    if (current == (old_owner & FUTEX_TID_MASK)) {
      switch (type) {
      case Type::Normal:
        break;
      case Type::ErrorChecking:
        return MutexError::DEADLOCK;
      case Type::Recursive:
        if (LIBC_UNLIKELY(*recursive_count ==
                          cpp::numeric_limits<size_t>::max()))
          return MutexError::OVERFLOW;
        *recursive_count += 1;
        return MutexError::NONE;
      }
    }

    return MutexError::BUSY;
  }
  LIBC_INLINE MutexError
  lock(cpp::optional<Futex::Timeout> timeout = cpp::nullopt,
       bool is_shared = false) {
    MutexError result = try_lock();
    if (result != MutexError::BUSY)
      return result;

#ifdef LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
    if (timeout)
      ensure_monotonicity(*timeout);
#endif

    int op = is_shared ? FUTEX_LOCK_PI2 : FUTEX_LOCK_PI2_PRIVATE;
    if (timeout && timeout->is_realtime())
      op |= FUTEX_CLOCK_REALTIME;
    for (;;) {
      ErrorOr<int> ret = linux_syscalls::futex(
          /*futex_addr=*/&owner,
          /*op=*/op,
          /*ignored=*/0,
          /*timeout=*/timeout ? &timeout->get_timespec() : nullptr,
          /*ignored=*/nullptr,
          /*ignored=*/0);

      if (ret.has_value())
        return MutexError::NONE;

      switch (ret.error()) {
      case EINTR:
        continue;
      case ETIMEDOUT:
        return MutexError::TIMEOUT;
      case EDEADLK:
        // indefinitely park the thread if type == Type::Normal.
        // use Futex::wait to avoid burning the CPU time
        while (type == Type::Normal)
          owner.wait(owner, /*timeout=*/cpp::nullopt, is_shared);
        return MutexError::DEADLOCK;
      default:
        return MutexError::BAD_LOCK_STATE;
      }
    }
  }
  LIBC_INLINE MutexError unlock(bool is_shared) {
    FutexWordType current = static_cast<FutexWordType>(internal::gettid());
    FutexWordType old_owner = current;

    if (LIBC_LIKELY(type == Type::Normal)) {
      if (LIBC_LIKELY(owner.compare_exchange_strong(old_owner, 0,
                                                    cpp::MemoryOrder::RELEASE,
                                                    cpp::MemoryOrder::RELAXED)))
        return MutexError::NONE;
    } else {
      old_owner = owner.load(cpp::MemoryOrder::RELAXED);
    }

    if (current != (old_owner & FUTEX_TID_MASK))
      return MutexError::UNLOCK_WITHOUT_LOCK;

    if (type == Type::Recursive && *recursive_count != 0) {
      *recursive_count -= 1;
      return MutexError::NONE;
    }

    if (old_owner == current && LIBC_LIKELY(owner.compare_exchange_strong(
                                    old_owner, 0, cpp::MemoryOrder::RELEASE,
                                    cpp::MemoryOrder::RELAXED)))
      return MutexError::NONE;

    int op = is_shared ? FUTEX_UNLOCK_PI : FUTEX_UNLOCK_PI_PRIVATE;
    ErrorOr<int> ret = linux_syscalls::futex(
        /*futex_addr=*/&owner,
        /*op=*/op,
        /*ignored=*/0,
        /*ignored=*/nullptr,
        /*ignored=*/nullptr,
        /*ignored=*/0);

    if (ret.has_value())
      return MutexError::NONE;

    switch (ret.error()) {
    case EPERM:
      return MutexError::UNLOCK_WITHOUT_LOCK;
    default:
      return MutexError::BAD_LOCK_STATE;
    }
  }
  LIBC_INLINE MutexError destroy() {
    FutexWordType old_owner = 0;
    if (owner.compare_exchange_strong(
            old_owner, cpp::numeric_limits<FutexWordType>::max(),
            cpp::MemoryOrder::RELAXED, cpp::MemoryOrder::RELAXED))
      return MutexError::NONE;
    return MutexError::BUSY;
  }
  LIBC_INLINE void reset() {
    owner.store(0);
    if (recursive_count)
      *recursive_count = 0;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_PI_MUTEX_H
