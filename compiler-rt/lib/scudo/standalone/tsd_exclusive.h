//===-- tsd_exclusive.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TSD_EXCLUSIVE_H_
#define SCUDO_TSD_EXCLUSIVE_H_

#include "tsd.h"

#include <pthread.h>

namespace scudo {

enum class ThreadState : u8 {
  NotInitialized = 0,
  Initialized,
  TornDown,
};

template <class Allocator> void teardownThread(void *Ptr);

template <class Allocator> struct TSDRegistryExT {
  void initLinkerInitialized(Allocator *Instance) {
    Instance->initLinkerInitialized();
    CHECK_EQ(pthread_key_create(&PThreadKey, teardownThread<Allocator>), 0);
    FallbackTSD = reinterpret_cast<TSD<Allocator> *>(
        map(nullptr, sizeof(TSD<Allocator>), "scudo:tsd"));
    FallbackTSD->initLinkerInitialized(Instance);
    Initialized = true;
  }
  void init(Allocator *Instance) {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized(Instance);
  }

  void unmapTestOnly() {
    unmap(reinterpret_cast<void *>(FallbackTSD), sizeof(TSD<Allocator>));
  }

  ALWAYS_INLINE void initThreadMaybe(Allocator *Instance, bool MinimalInit) {
    if (LIKELY(State != ThreadState::NotInitialized))
      return;
    initThread(Instance, MinimalInit);
  }

  ALWAYS_INLINE TSD<Allocator> *getTSDAndLock(bool *UnlockRequired) {
    if (LIKELY(State == ThreadState::Initialized)) {
      *UnlockRequired = false;
      return &ThreadTSD;
    }
    DCHECK(FallbackTSD);
    FallbackTSD->lock();
    *UnlockRequired = true;
    return FallbackTSD;
  }

private:
  void initOnceMaybe(Allocator *Instance) {
    ScopedLock L(Mutex);
    if (LIKELY(Initialized))
      return;
    initLinkerInitialized(Instance); // Sets Initialized.
  }

  // Using minimal initialization allows for global initialization while keeping
  // the thread specific structure untouched. The fallback structure will be
  // used instead.
  NOINLINE void initThread(Allocator *Instance, bool MinimalInit) {
    initOnceMaybe(Instance);
    if (UNLIKELY(MinimalInit))
      return;
    CHECK_EQ(
        pthread_setspecific(PThreadKey, reinterpret_cast<void *>(Instance)), 0);
    ThreadTSD.initLinkerInitialized(Instance);
    State = ThreadState::Initialized;
  }

  pthread_key_t PThreadKey;
  bool Initialized;
  TSD<Allocator> *FallbackTSD;
  HybridMutex Mutex;
  static THREADLOCAL ThreadState State;
  static THREADLOCAL TSD<Allocator> ThreadTSD;

  friend void teardownThread<Allocator>(void *Ptr);
};

template <class Allocator>
THREADLOCAL TSD<Allocator> TSDRegistryExT<Allocator>::ThreadTSD;
template <class Allocator>
THREADLOCAL ThreadState TSDRegistryExT<Allocator>::State;

template <class Allocator> void teardownThread(void *Ptr) {
  typedef TSDRegistryExT<Allocator> TSDRegistryT;
  Allocator *Instance = reinterpret_cast<Allocator *>(Ptr);
  // The glibc POSIX thread-local-storage deallocation routine calls user
  // provided destructors in a loop of PTHREAD_DESTRUCTOR_ITERATIONS.
  // We want to be called last since other destructors might call free and the
  // like, so we wait until PTHREAD_DESTRUCTOR_ITERATIONS before draining the
  // quarantine and swallowing the cache.
  if (TSDRegistryT::ThreadTSD.DestructorIterations > 1) {
    TSDRegistryT::ThreadTSD.DestructorIterations--;
    // If pthread_setspecific fails, we will go ahead with the teardown.
    if (LIKELY(pthread_setspecific(Instance->getTSDRegistry()->PThreadKey,
                                   Ptr) == 0))
      return;
  }
  TSDRegistryT::ThreadTSD.commitBack(Instance);
  TSDRegistryT::State = ThreadState::TornDown;
}

} // namespace scudo

#endif // SCUDO_TSD_EXCLUSIVE_H_
