//===-- mutex.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_MUTEX_H_
#define SCUDO_MUTEX_H_

#include "atomic_helpers.h"
#include "common.h"

#include <string.h>

#if SCUDO_FUCHSIA
#include <lib/sync/mutex.h> // for sync_mutex_t
#endif

namespace scudo {

class HybridMutex {
public:
  void init() { M = {}; }
  bool tryLock();
  NOINLINE void lock() {
    if (LIKELY(tryLock()))
      return;
      // The compiler may try to fully unroll the loop, ending up in a
      // NumberOfTries*NumberOfYields block of pauses mixed with tryLocks. This
      // is large, ugly and unneeded, a compact loop is better for our purpose
      // here. Use a pragma to tell the compiler not to unroll the loop.
#ifdef __clang__
#pragma nounroll
#endif
    for (u8 I = 0U; I < NumberOfTries; I++) {
      yieldProcessor(NumberOfYields);
      if (tryLock())
        return;
    }
    lockSlow();
  }
  void unlock();

private:
  static constexpr u8 NumberOfTries = 8U;
  static constexpr u8 NumberOfYields = 8U;

#if SCUDO_LINUX
  atomic_u32 M;
#elif SCUDO_FUCHSIA
  sync_mutex_t M;
#endif

  void lockSlow();
};

class ScopedLock {
public:
  explicit ScopedLock(HybridMutex &M) : Mutex(M) { Mutex.lock(); }
  ~ScopedLock() { Mutex.unlock(); }

private:
  HybridMutex &Mutex;

  ScopedLock(const ScopedLock &) = delete;
  void operator=(const ScopedLock &) = delete;
};

} // namespace scudo

#endif // SCUDO_MUTEX_H_
