//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-threads

// GCC doesn't have thread safety attributes
// UNSUPPORTED: gcc

// ADDITIONAL_COMPILE_FLAGS: -Wthread-safety

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
    ++data;     // expected-warning {{writing variable 'data' requires holding shared_mutex 'm' exclusively}}
    m.unlock_shared();
  }
  {
    if (m.try_lock_shared()) {
      read(data); // ok
      ++data;     // expected-warning {{writing variable 'data' requires holding shared_mutex 'm' exclusively}}
      m.unlock_shared();
    }
  }
}
