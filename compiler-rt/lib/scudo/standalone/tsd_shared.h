//===-- tsd_shared.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TSD_SHARED_H_
#define SCUDO_TSD_SHARED_H_

#include "linux.h" // for getAndroidTlsPtr()
#include "tsd.h"

namespace scudo {

template <class Allocator, u32 MaxTSDCount> struct TSDRegistrySharedT {
  void initLinkerInitialized(Allocator *Instance) {
    Instance->initLinkerInitialized();
    CHECK_EQ(pthread_key_create(&PThreadKey, nullptr), 0); // For non-TLS
    const u32 NumberOfCPUs = getNumberOfCPUs();
    NumberOfTSDs =
        (NumberOfCPUs == 0) ? MaxTSDCount : Min(NumberOfCPUs, MaxTSDCount);
    TSDs = reinterpret_cast<TSD<Allocator> *>(
        map(nullptr, sizeof(TSD<Allocator>) * NumberOfTSDs, "scudo:tsd"));
    for (u32 I = 0; I < NumberOfTSDs; I++)
      TSDs[I].initLinkerInitialized(Instance);
    // Compute all the coprimes of NumberOfTSDs. This will be used to walk the
    // array of TSDs in a random order. For details, see:
    // https://lemire.me/blog/2017/09/18/visiting-all-values-in-an-array-exactly-once-in-random-order/
    for (u32 I = 0; I < NumberOfTSDs; I++) {
      u32 A = I + 1;
      u32 B = NumberOfTSDs;
      // Find the GCD between I + 1 and NumberOfTSDs. If 1, they are coprimes.
      while (B != 0) {
        const u32 T = A;
        A = B;
        B = T % B;
      }
      if (A == 1)
        CoPrimes[NumberOfCoPrimes++] = I + 1;
    }
    Initialized = true;
  }
  void init(Allocator *Instance) {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized(Instance);
  }

  void unmapTestOnly() {
    unmap(reinterpret_cast<void *>(TSDs),
          sizeof(TSD<Allocator>) * NumberOfTSDs);
    setCurrentTSD(nullptr);
    pthread_key_delete(PThreadKey);
  }

  ALWAYS_INLINE void initThreadMaybe(Allocator *Instance,
                                     UNUSED bool MinimalInit) {
    if (LIKELY(getCurrentTSD()))
      return;
    initThread(Instance);
  }

  ALWAYS_INLINE TSD<Allocator> *getTSDAndLock(bool *UnlockRequired) {
    TSD<Allocator> *TSD = getCurrentTSD();
    DCHECK(TSD);
    *UnlockRequired = true;
    // Try to lock the currently associated context.
    if (TSD->tryLock())
      return TSD;
    // If that fails, go down the slow path.
    return getTSDAndLockSlow(TSD);
  }

  void disable() {
    Mutex.lock();
    for (u32 I = 0; I < NumberOfTSDs; I++)
      TSDs[I].lock();
  }

  void enable() {
    for (s32 I = NumberOfTSDs - 1; I >= 0; I--)
      TSDs[I].unlock();
    Mutex.unlock();
  }

private:
  ALWAYS_INLINE void setCurrentTSD(TSD<Allocator> *CurrentTSD) {
#if _BIONIC
    *getAndroidTlsPtr() = reinterpret_cast<uptr>(CurrentTSD);
#elif SCUDO_LINUX
    ThreadTSD = CurrentTSD;
#else
    CHECK_EQ(
        pthread_setspecific(PThreadKey, reinterpret_cast<void *>(CurrentTSD)),
        0);
#endif
  }

  ALWAYS_INLINE TSD<Allocator> *getCurrentTSD() {
#if _BIONIC
    return reinterpret_cast<TSD<Allocator> *>(*getAndroidTlsPtr());
#elif SCUDO_LINUX
    return ThreadTSD;
#else
    return reinterpret_cast<TSD<Allocator> *>(pthread_getspecific(PThreadKey));
#endif
  }

  void initOnceMaybe(Allocator *Instance) {
    ScopedLock L(Mutex);
    if (LIKELY(Initialized))
      return;
    initLinkerInitialized(Instance); // Sets Initialized.
  }

  NOINLINE void initThread(Allocator *Instance) {
    initOnceMaybe(Instance);
    // Initial context assignment is done in a plain round-robin fashion.
    const u32 Index = atomic_fetch_add(&CurrentIndex, 1U, memory_order_relaxed);
    setCurrentTSD(&TSDs[Index % NumberOfTSDs]);
    Instance->callPostInitCallback();
  }

  NOINLINE TSD<Allocator> *getTSDAndLockSlow(TSD<Allocator> *CurrentTSD) {
    if (MaxTSDCount > 1U && NumberOfTSDs > 1U) {
      // Use the Precedence of the current TSD as our random seed. Since we are
      // in the slow path, it means that tryLock failed, and as a result it's
      // very likely that said Precedence is non-zero.
      const u32 R = static_cast<u32>(CurrentTSD->getPrecedence());
      const u32 Inc = CoPrimes[R % NumberOfCoPrimes];
      u32 Index = R % NumberOfTSDs;
      uptr LowestPrecedence = UINTPTR_MAX;
      TSD<Allocator> *CandidateTSD = nullptr;
      // Go randomly through at most 4 contexts and find a candidate.
      for (u32 I = 0; I < Min(4U, NumberOfTSDs); I++) {
        if (TSDs[Index].tryLock()) {
          setCurrentTSD(&TSDs[Index]);
          return &TSDs[Index];
        }
        const uptr Precedence = TSDs[Index].getPrecedence();
        // A 0 precedence here means another thread just locked this TSD.
        if (Precedence && Precedence < LowestPrecedence) {
          CandidateTSD = &TSDs[Index];
          LowestPrecedence = Precedence;
        }
        Index += Inc;
        if (Index >= NumberOfTSDs)
          Index -= NumberOfTSDs;
      }
      if (CandidateTSD) {
        CandidateTSD->lock();
        setCurrentTSD(CandidateTSD);
        return CandidateTSD;
      }
    }
    // Last resort, stick with the current one.
    CurrentTSD->lock();
    return CurrentTSD;
  }

  pthread_key_t PThreadKey;
  atomic_u32 CurrentIndex;
  u32 NumberOfTSDs;
  TSD<Allocator> *TSDs;
  u32 NumberOfCoPrimes;
  u32 CoPrimes[MaxTSDCount];
  bool Initialized;
  HybridMutex Mutex;
#if SCUDO_LINUX && !_BIONIC
  static THREADLOCAL TSD<Allocator> *ThreadTSD;
#endif
};

#if SCUDO_LINUX && !_BIONIC
template <class Allocator, u32 MaxTSDCount>
THREADLOCAL TSD<Allocator>
    *TSDRegistrySharedT<Allocator, MaxTSDCount>::ThreadTSD;
#endif

} // namespace scudo

#endif // SCUDO_TSD_SHARED_H_
