//===-- A platform independent abstraction layer for cond vars --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC___SUPPORT_SRC_THREADS_LINUX_CNDVAR_H
#define LLVM_LIBC___SUPPORT_SRC_THREADS_LINUX_CNDVAR_H

#include "hdr/stdint_proxy.h" // uint32_t
#include "src/__support/CPP/optional.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/futex_utils.h" // Futex
#include "src/__support/threads/mutex.h"             // Mutex
#include "src/__support/threads/raw_mutex.h"         // RawMutex
#include "src/__support/time/abs_timeout.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: this implementation seems to be contention heavy. We probably want to
// use a more efficient implementation.
// TODO: thread local stack is not accessible in shared mode. We need to either
// fallback to single futex word under shared memory or use a totally
// different implementation.
// TODO: node clean up needs to be registered into pthread cancellation point
// cleanup routine once that functionality is available.
class CndVar {
  enum CndWaiterStatus : uint32_t {
    WS_Waiting = 0xE,
    WS_Signalled = 0x5,
  };

  struct WQNode {
    WQNode *prev;
    WQNode *next;
  };

  struct CndWaiter : WQNode {
    Futex futex_word = WS_Waiting;
  };

  WQNode waitq;
  RawMutex qmtx;

public:
  enum class Result {
    Success,
    MutexError,
    Timeout,
  };

  LIBC_INLINE static int init(CndVar *cv) {
    cv->waitq.prev = cv->waitq.next = nullptr;
    RawMutex::init(&cv->qmtx);
    return 0;
  }

  LIBC_INLINE static void destroy(CndVar *cv) {
    cv->waitq.prev = cv->waitq.next = nullptr;
  }

  // Returns 0 on success, -1 on error.
  Result wait(Mutex *m,
              cpp::optional<internal::AbsTimeout> timeout = cpp::nullopt);
  void notify_one();
  void broadcast();

private:
  void ensure_cyclic_queue();
  void push_back(CndWaiter *w);
  CndWaiter *pop_front();
  void remove(CndWaiter *w);
  CndWaiter *take_all();
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC___SUPPORT_SRC_THREADS_LINUX_CNDVAR_H
