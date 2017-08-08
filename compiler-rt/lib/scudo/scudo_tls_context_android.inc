//===-- scudo_tls_context_android.inc ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Android specific base thread context definition.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TLS_CONTEXT_ANDROID_INC_
#define SCUDO_TLS_CONTEXT_ANDROID_INC_

#ifndef SCUDO_TLS_H_
# error "This file must be included inside scudo_tls.h."
#endif  // SCUDO_TLS_H_

#if SANITIZER_LINUX && SANITIZER_ANDROID

struct ScudoThreadContextPlatform {
  INLINE bool tryLock() {
    if (Mutex.TryLock()) {
      atomic_store_relaxed(&SlowLockPrecedence, 0);
      return true;
    }
    if (atomic_load_relaxed(&SlowLockPrecedence) == 0)
      atomic_store_relaxed(&SlowLockPrecedence, NanoTime());
    return false;
  }

  INLINE void lock() {
    Mutex.Lock();
    atomic_store_relaxed(&SlowLockPrecedence, 0);
  }

  INLINE void unlock() {
    Mutex.Unlock();
  }

  INLINE u64 getSlowLockPrecedence() {
    return atomic_load_relaxed(&SlowLockPrecedence);
  }

 private:
  StaticSpinMutex Mutex;
  atomic_uint64_t SlowLockPrecedence;
};

#endif  // SANITIZER_LINUX && SANITIZER_ANDROID

#endif  // SCUDO_TLS_CONTEXT_ANDROID_INC_
