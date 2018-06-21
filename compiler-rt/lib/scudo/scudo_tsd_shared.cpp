//===-- scudo_tsd_shared.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo shared TSD implementation.
///
//===----------------------------------------------------------------------===//

#include "scudo_tsd.h"

#if !SCUDO_TSD_EXCLUSIVE

namespace __scudo {

static pthread_once_t GlobalInitialized = PTHREAD_ONCE_INIT;
pthread_key_t PThreadKey;

static atomic_uint32_t CurrentIndex;
static ScudoTSD *TSDs;
static u32 NumberOfTSDs;

static void initOnce() {
  CHECK_EQ(pthread_key_create(&PThreadKey, NULL), 0);
  initScudo();
  NumberOfTSDs = Min(Max(1U, GetNumberOfCPUsCached()),
                     static_cast<u32>(SCUDO_SHARED_TSD_POOL_SIZE));
  TSDs = reinterpret_cast<ScudoTSD *>(
      MmapOrDie(sizeof(ScudoTSD) * NumberOfTSDs, "ScudoTSDs"));
  for (u32 i = 0; i < NumberOfTSDs; i++)
    TSDs[i].init(/*Shared=*/true);
}

ALWAYS_INLINE void setCurrentTSD(ScudoTSD *TSD) {
#if SANITIZER_ANDROID
  *get_android_tls_ptr() = reinterpret_cast<uptr>(TSD);
#else
  CHECK_EQ(pthread_setspecific(PThreadKey, reinterpret_cast<void *>(TSD)), 0);
#endif  // SANITIZER_ANDROID
}

void initThread(bool MinimalInit) {
  pthread_once(&GlobalInitialized, initOnce);
  // Initial context assignment is done in a plain round-robin fashion.
  u32 Index = atomic_fetch_add(&CurrentIndex, 1, memory_order_relaxed);
  setCurrentTSD(&TSDs[Index % NumberOfTSDs]);
}

ScudoTSD *getTSDAndLockSlow() {
  ScudoTSD *TSD;
  if (NumberOfTSDs > 1) {
    // Go through all the contexts and find the first unlocked one.
    for (u32 i = 0; i < NumberOfTSDs; i++) {
      TSD = &TSDs[i];
      if (TSD->tryLock()) {
        setCurrentTSD(TSD);
        return TSD;
      }
    }
    // No luck, find the one with the lowest Precedence, and slow lock it.
    u64 LowestPrecedence = UINT64_MAX;
    for (u32 i = 0; i < NumberOfTSDs; i++) {
      u64 Precedence = TSDs[i].getPrecedence();
      if (Precedence && Precedence < LowestPrecedence) {
        TSD = &TSDs[i];
        LowestPrecedence = Precedence;
      }
    }
    if (LIKELY(LowestPrecedence != UINT64_MAX)) {
      TSD->lock();
      setCurrentTSD(TSD);
      return TSD;
    }
  }
  // Last resort, stick with the current one.
  TSD = getCurrentTSD();
  TSD->lock();
  return TSD;
}

}  // namespace __scudo

#endif  // !SCUDO_TSD_EXCLUSIVE
