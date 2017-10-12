//===-- scudo_tsd_shared.inc ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

ALWAYS_INLINE ScudoTSD* getCurrentTSD() {
  return reinterpret_cast<ScudoTSD *>(*get_android_tls_ptr());
}

ALWAYS_INLINE void initThreadMaybe(bool MinimalInit = false) {
  if (LIKELY(getCurrentTSD()))
    return;
  initThread(MinimalInit);
}

ScudoTSD *getTSDAndLockSlow();

ALWAYS_INLINE ScudoTSD *getTSDAndLock() {
  ScudoTSD *TSD = getCurrentTSD();
  CHECK(TSD && "No TSD associated with the current thread!");
  // Try to lock the currently associated context.
  if (TSD->tryLock())
    return TSD;
  // If it failed, go the slow path.
  return getTSDAndLockSlow();
}

#endif  // !SCUDO_TSD_EXCLUSIVE
