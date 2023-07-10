//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-threads
// UNSUPPORTED: availability-shared_mutex-missing
// REQUIRES: thread-safety
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS

// On Windows Clang bugs out when both __declspec and __attribute__ are present,
// the processing goes awry preventing the definition of the types.
// XFAIL: msvc

// <shared_mutex>
//
// class shared_mutex;
//
// void lock();
// bool try_lock();
// void unlock();
//
// void lock_shared();
// bool try_lock_shared();
// void unlock_shared();

#include <shared_mutex>

std::shared_mutex m;
int data __attribute__((guarded_by(m))) = 0;
void read(int);

void f() {
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

  // Shared locking
  {
    m.lock_shared();
    read(data); // ok
    ++data; // expected-error {{writing variable 'data' requires holding shared_mutex 'm' exclusively}}
    m.unlock_shared();
  }
  {
    if (m.try_lock_shared()) {
      read(data); // ok
      ++data; // expected-error {{writing variable 'data' requires holding shared_mutex 'm' exclusively}}
      m.unlock_shared();
    }
  }
}
