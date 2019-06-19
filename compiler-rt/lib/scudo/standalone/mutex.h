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

namespace scudo {

class StaticSpinMutex {
public:
  void init() { atomic_store_relaxed(&State, 0); }

  void lock() {
    if (tryLock())
      return;
    lockSlow();
  }

  bool tryLock() {
    return atomic_exchange(&State, 1, memory_order_acquire) == 0;
  }

  void unlock() { atomic_store(&State, 0, memory_order_release); }

  void checkLocked() { CHECK_EQ(atomic_load_relaxed(&State), 1); }

private:
  atomic_u8 State;

  void NOINLINE lockSlow() {
    for (u32 I = 0;; I++) {
      if (I < 10)
        yieldProcessor(10);
      else
        yieldPlatform();
      if (atomic_load_relaxed(&State) == 0 &&
          atomic_exchange(&State, 1, memory_order_acquire) == 0)
        return;
    }
  }
};

class SpinMutex : public StaticSpinMutex {
public:
  SpinMutex() { init(); }

private:
  SpinMutex(const SpinMutex &) = delete;
  void operator=(const SpinMutex &) = delete;
};

class BlockingMutex {
public:
  explicit constexpr BlockingMutex(LinkerInitialized) : OpaqueStorage{} {}
  BlockingMutex() { memset(this, 0, sizeof(*this)); }
  void lock();
  void unlock();
  void checkLocked() {
    atomic_u32 *M = reinterpret_cast<atomic_u32 *>(&OpaqueStorage);
    CHECK_NE(MtxUnlocked, atomic_load_relaxed(M));
  }

private:
  enum MutexState { MtxUnlocked = 0, MtxLocked = 1, MtxSleeping = 2 };
  uptr OpaqueStorage[1];
};

template <typename MutexType> class GenericScopedLock {
public:
  explicit GenericScopedLock(MutexType *M) : Mutex(M) { Mutex->lock(); }
  ~GenericScopedLock() { Mutex->unlock(); }

private:
  MutexType *Mutex;

  GenericScopedLock(const GenericScopedLock &) = delete;
  void operator=(const GenericScopedLock &) = delete;
};

typedef GenericScopedLock<StaticSpinMutex> SpinMutexLock;
typedef GenericScopedLock<BlockingMutex> BlockingMutexLock;

} // namespace scudo

#endif // SCUDO_MUTEX_H_
