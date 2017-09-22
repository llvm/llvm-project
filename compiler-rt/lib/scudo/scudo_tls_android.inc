//===-- scudo_tls_android.inc -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo thread local structure fastpath functions implementation for Android.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TLS_ANDROID_H_
#define SCUDO_TLS_ANDROID_H_

#ifndef SCUDO_TLS_H_
# error "This file must be included inside scudo_tls.h."
#endif  // SCUDO_TLS_H_

#if SANITIZER_LINUX && SANITIZER_ANDROID

ALWAYS_INLINE void initThreadMaybe(bool MinimalInit = false) {
  if (LIKELY(*get_android_tls_ptr()))
    return;
  initThread(MinimalInit);
}

ScudoTSD *getTSDAndLockSlow();

ALWAYS_INLINE ScudoTSD *getTSDAndLock() {
  ScudoTSD *TSD = reinterpret_cast<ScudoTSD *>(*get_android_tls_ptr());
  CHECK(TSD);
  // Try to lock the currently associated context.
  if (TSD->tryLock())
    return TSD;
  // If it failed, go the slow path.
  return getTSDAndLockSlow();
}

#endif  // SANITIZER_LINUX && SANITIZER_ANDROID

#endif  // SCUDO_TLS_ANDROID_H_
