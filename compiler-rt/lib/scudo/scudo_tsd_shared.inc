//===-- scudo_tsd_shared.inc ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Scudo shared TSD fastpath functions implementation.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TSD_H_
# error "This file must be included inside scudo_tsd.h."
#endif  // SCUDO_TSD_H_

#if !SCUDO_TSD_EXCLUSIVE

extern pthread_key_t PThreadKey;

#if SANITIZER_LINUX && !SANITIZER_ANDROID
__attribute__((tls_model("initial-exec")))
extern THREADLOCAL ScudoTSD *CurrentTSD;
#endif

ALWAYS_INLINE ScudoTSD* getCurrentTSD() {
#if SANITIZER_ANDROID
  return reinterpret_cast<ScudoTSD *>(*get_android_tls_ptr());
#elif SANITIZER_LINUX
  return CurrentTSD;
#else
  return reinterpret_cast<ScudoTSD *>(pthread_getspecific(PThreadKey));
#endif  // SANITIZER_ANDROID
}

ALWAYS_INLINE void initThreadMaybe(bool MinimalInit = false) {
  if (LIKELY(getCurrentTSD()))
    return;
  initThread(MinimalInit);
}

ScudoTSD *getTSDAndLockSlow(ScudoTSD *TSD);

ALWAYS_INLINE ScudoTSD *
getTSDAndLock(bool *UnlockRequired) SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  ScudoTSD *TSD = getCurrentTSD();
  DCHECK(TSD && "No TSD associated with the current thread!");
  *UnlockRequired = true;
  // Try to lock the currently associated context.
  if (TSD->tryLock())
    return TSD;
  // If it failed, go the slow path.
  return getTSDAndLockSlow(TSD);
}

#endif  // !SCUDO_TSD_EXCLUSIVE
