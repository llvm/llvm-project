//===-- runtime/lock.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Wraps a mutex

#ifndef FORTRAN_RUNTIME_LOCK_H_
#define FORTRAN_RUNTIME_LOCK_H_

#include "terminator.h"
#include "tools.h"

// Avoid <mutex> if possible to avoid introduction of C++ runtime
// library dependence.
#ifndef _WIN32
#define USE_PTHREADS 1
#else
#undef USE_PTHREADS
#endif

#if USE_PTHREADS
#include <pthread.h>
#elif defined(_WIN32)
// Do not define macros for "min" and "max"
#define NOMINMAX
#include <windows.h>
#else
#include <mutex>
#endif

namespace Fortran::runtime {

class Lock {
public:
#if RT_USE_PSEUDO_LOCK
  // No lock implementation, e.g. for using together
  // with RT_USE_PSEUDO_FILE_UNIT.
  // The users of Lock class may use it under
  // USE_PTHREADS and otherwise, so it has to provide
  // all the interfaces.
  void Take() {}
  bool Try() { return true; }
  void Drop() {}
  bool TakeIfNoDeadlock() { return true; }
#elif USE_PTHREADS
  Lock() { pthread_mutex_init(&mutex_, nullptr); }
  ~Lock() { pthread_mutex_destroy(&mutex_); }
  void Take() {
    while (pthread_mutex_lock(&mutex_)) {
    }
    holder_ = pthread_self();
    isBusy_ = true;
  }
  bool TakeIfNoDeadlock() {
    if (isBusy_) {
      auto thisThread{pthread_self()};
      if (pthread_equal(thisThread, holder_)) {
        return false;
      }
    }
    Take();
    return true;
  }
  bool Try() { return pthread_mutex_trylock(&mutex_) == 0; }
  void Drop() {
    isBusy_ = false;
    pthread_mutex_unlock(&mutex_);
  }
#elif defined(_WIN32)
  Lock() { InitializeCriticalSection(&cs_); }
  ~Lock() { DeleteCriticalSection(&cs_); }
  void Take() { EnterCriticalSection(&cs_); }
  bool Try() { return TryEnterCriticalSection(&cs_); }
  void Drop() { LeaveCriticalSection(&cs_); }
#else
  void Take() { mutex_.lock(); }
  bool Try() { return mutex_.try_lock(); }
  void Drop() { mutex_.unlock(); }
#endif

  void CheckLocked(const Terminator &terminator) {
    if (Try()) {
      Drop();
      terminator.Crash("Lock::CheckLocked() failed");
    }
  }

private:
#if RT_USE_PSEUDO_FILE_UNIT
  // No state.
#elif USE_PTHREADS
  pthread_mutex_t mutex_{};
  volatile bool isBusy_{false};
  volatile pthread_t holder_;
#elif defined(_WIN32)
  CRITICAL_SECTION cs_;
#else
  std::mutex mutex_;
#endif
};

class CriticalSection {
public:
  explicit CriticalSection(Lock &lock) : lock_{lock} { lock_.Take(); }
  ~CriticalSection() { lock_.Drop(); }

private:
  Lock &lock_;
};
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_LOCK_H_
