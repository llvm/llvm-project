//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: no-threads
// UNSUPPORTED: availability-shared_mutex-missing
// REQUIRES: thread-safety
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS

// On Windows Clang bugs out when both __declspec and __attribute__ are present,
// the processing goes awry preventing the definition of the types.
// XFAIL: msvc

// <shared_mutex>
//
// class shared_timed_mutex;
//
// void lock();
// bool try_lock();
// bool try_lock_for(const std::chrono::duration<Rep, Period>&);
// bool try_lock_until(const std::chrono::time_point<Clock, Duration>&);
// void unlock();
//
// void lock_shared();
// bool try_lock_shared();
// bool try_lock_shared_for(const std::chrono::duration<Rep, Period>&);
// bool try_lock_shared_until(const std::chrono::time_point<Clock, Duration>&);
// void unlock_shared();

#include <chrono>
#include <shared_mutex>

std::shared_timed_mutex m;
int data __attribute__((guarded_by(m))) = 0;
void read(int);

void f(std::chrono::time_point<std::chrono::steady_clock> tp, std::chrono::milliseconds d) {
  // Exclusive locking
  {
    m.lock();
    ++data; // ok
    m.unlock();
  }
  {
    if (m.try_lock()) {
      ++data; // ok
      m.unlock();
    }
  }
  {
    if (m.try_lock_for(d)) {
      ++data; // ok
      m.unlock();
    }
  }
  {
    if (m.try_lock_until(tp)) {
      ++data; // ok
      m.unlock();
    }
  }

  // Shared locking
  {
    m.lock_shared();
    read(data); // ok
    ++data; // expected-error {{writing variable 'data' requires holding shared_timed_mutex 'm' exclusively}}
    m.unlock_shared();
  }
  {
    if (m.try_lock_shared()) {
      read(data); // ok
      ++data; // expected-error {{writing variable 'data' requires holding shared_timed_mutex 'm' exclusively}}
      m.unlock_shared();
    }
  }
  {
    if (m.try_lock_shared_for(d)) {
      read(data); // ok
      ++data; // expected-error {{writing variable 'data' requires holding shared_timed_mutex 'm' exclusively}}
      m.unlock_shared();
    }
  }
  {
    if (m.try_lock_shared_until(tp)) {
      read(data); // ok
      ++data; // expected-error {{writing variable 'data' requires holding shared_timed_mutex 'm' exclusively}}
      m.unlock_shared();
    }
  }
}
