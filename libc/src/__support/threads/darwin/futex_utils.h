//===--- Futex utils for Darwin -----------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_DARWIN_FUTEX_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_DARWIN_FUTEX_UTILS_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/time/abs_timeout.h"
#include "src/__support/time/clock_conversion.h"

#include <os/os_sync_wait_on_address.h>

namespace LIBC_NAMESPACE_DECL {

using FutexWordType = uint32_t;

struct Futex : public cpp::Atomic<FutexWordType> {
  using cpp::Atomic<FutexWordType>::Atomic;
  using Timeout = internal::AbsTimeout;

  // The Darwin futex API does not return a value on timeout, so we have to
  // check for it manually. This means we can't use the return value to
  // distinguish between a timeout and a successful wake-up.
  int wait(FutexWordType val, cpp::optional<Timeout> timeout, bool) {
    if (timeout) {
      struct timespec now;
      clock_gettime(timeout->is_realtime() ? CLOCK_REALTIME : CLOCK_MONOTONIC,
                    &now);
      const timespec &target_ts = timeout->get_timespec();

      if (now.tv_sec > target_ts.tv_sec ||
          (now.tv_sec == target_ts.tv_sec && now.tv_nsec >= target_ts.tv_nsec))
        return ETIMEDOUT;
    }

    os_sync_wait_on_address(reinterpret_cast<void *>(this),
                            static_cast<uint64_t>(val), sizeof(FutexWordType),
                            OS_SYNC_WAIT_ON_ADDRESS_NONE);
    return 0;
  }

  void notify_one(bool) {
    os_sync_wake_by_address_any(reinterpret_cast<void *>(this),
                                sizeof(FutexWordType),
                                OS_SYNC_WAKE_BY_ADDRESS_NONE);
  }

  void notify_all(bool) {
    // os_sync_wake_by_address_all is not available, so we use notify_one.
    // This is not ideal, but it's the best we can do with the available API.
    os_sync_wake_by_address_any(reinterpret_cast<void *>(this),
                                sizeof(FutexWordType),
                                OS_SYNC_WAKE_BY_ADDRESS_NONE);
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_DARWIN_FUTEX_UTILS_H
