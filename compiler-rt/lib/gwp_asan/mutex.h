//===-- mutex.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_MUTEX_H_
#define GWP_ASAN_MUTEX_H_

#ifdef __unix__
#include <pthread.h>
#else
#error "GWP-ASan is not supported on this platform."
#endif

namespace gwp_asan {
class Mutex {
public:
  constexpr Mutex() = default;
  ~Mutex() = default;
  Mutex(const Mutex &) = delete;
  Mutex &operator=(const Mutex &) = delete;
  // Lock the mutex.
  void lock();
  // Nonblocking trylock of the mutex. Returns true if the lock was acquired.
  bool tryLock();
  // Unlock the mutex.
  void unlock();

private:
#ifdef __unix__
  pthread_mutex_t Mu = PTHREAD_MUTEX_INITIALIZER;
#endif // defined(__unix__)
};

class ScopedLock {
public:
  explicit ScopedLock(Mutex &Mx) : Mu(Mx) { Mu.lock(); }
  ~ScopedLock() { Mu.unlock(); }
  ScopedLock(const ScopedLock &) = delete;
  ScopedLock &operator=(const ScopedLock &) = delete;

private:
  Mutex &Mu;
};
} // namespace gwp_asan

#endif // GWP_ASAN_MUTEX_H_
