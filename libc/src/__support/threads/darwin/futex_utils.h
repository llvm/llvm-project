//===--- Futex utils for Darwin ----------------------------------*- C++-*-===//
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
#include "src/__support/time/units.h"

#include <os/os_sync_wait_on_address.h>

namespace LIBC_NAMESPACE_DECL {

using FutexWordType = uint32_t;

struct Futex : public cpp::Atomic<FutexWordType> {
  using cpp::Atomic<FutexWordType>::Atomic;
  using Timeout = internal::AbsTimeout;

  LIBC_INLINE long wait(FutexWordType val, cpp::optional<Timeout> timeout,
                        bool /* is_shared */) {
    // TODO(bojle): consider using OS_SYNC_WAIT_ON_ADDRESS_SHARED to sync
    // betweeen processes. Catch: it is recommended to only be used by shared
    // processes, not threads of a same process.

    for (;;) {
      if (this->load(cpp::MemoryOrder::RELAXED) != val)
        return 0;
      long ret = 0;
      if (timeout) {
        // Assuming, OS_CLOCK_MACH_ABSOLUTE_TIME is equivalent to CLOCK_REALTIME
        using namespace time_units;
        uint64_t tnsec = timeout->get_timespec().tv_sec * 1_s_ns +
                         timeout->get_timespec().tv_nsec;
        ret = os_sync_wait_on_address_with_timeout(
            reinterpret_cast<void *>(this), static_cast<uint64_t>(val),
            sizeof(FutexWordType), OS_SYNC_WAIT_ON_ADDRESS_NONE,
            OS_CLOCK_MACH_ABSOLUTE_TIME, tnsec);
      } else {
        ret = os_sync_wait_on_address(
            reinterpret_cast<void *>(this), static_cast<uint64_t>(val),
            sizeof(FutexWordType), OS_SYNC_WAIT_ON_ADDRESS_NONE);
      }
      if ((ret < 0) && (errno == ETIMEDOUT))
        return -ETIMEDOUT;
      // case when os_sync returns early with an error. retry.
      if ((ret < 0) && ((errno == EINTR) || (errno == EFAULT))) {
        continue;
      }
      return ret;
    }
  }

  LIBC_INLINE long notify_one(bool /* is_shared */) {
    // TODO(bojle): deal with is_shared
    return os_sync_wake_by_address_any(reinterpret_cast<void *>(this),
                                       sizeof(FutexWordType),
                                       OS_SYNC_WAKE_BY_ADDRESS_NONE);
  }

  LIBC_INLINE long notify_all(bool /* is_shared */) {
    // TODO(bojle): deal with is_shared
    return os_sync_wake_by_address_all(reinterpret_cast<void *>(this),
                                       sizeof(FutexWordType),
                                       OS_SYNC_WAKE_BY_ADDRESS_NONE);
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_DARWIN_FUTEX_UTILS_H
