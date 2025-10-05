//===--- Futex Wrapper ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_FUTEX_TIMEOUT_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_FUTEX_TIMEOUT_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/futex_utils.h"
#include "src/__support/time/linux/abs_timeout.h"

namespace LIBC_NAMESPACE_DECL {
class TimedFutex : public Futex {
public:
  using Timeout = internal::AbsTimeout;
  using Futex::Futex;
  using Futex::operator=;
  LIBC_INLINE long wait(FutexWordType expected,
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

      long ret = syscall_impl<long>(
          /* syscall number */ FUTEX_SYSCALL_ID,
          /* futex address */ this,
          /* futex operation  */ op,
          /* expected value */ expected,
          /* timeout */ timeout ? &timeout->get_timespec() : nullptr,
          /* ignored */ nullptr,
          /* bitset */ FUTEX_BITSET_MATCH_ANY);

      // continue waiting if interrupted; otherwise return the result
      // which should normally be 0 or -ETIMEOUT
      if (ret == -EINTR)
        continue;

      return ret;
    }
  }
};

static_assert(__is_standard_layout(TimedFutex),
              "Futex must be a standard layout type.");
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_FUTEX_TIMEOUT_H
