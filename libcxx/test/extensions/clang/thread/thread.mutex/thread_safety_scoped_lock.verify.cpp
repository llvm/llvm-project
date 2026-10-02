//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads, c++03, c++11, c++14

// <mutex>

// GCC doesn't have thread safety attributes
// UNSUPPORTED: gcc

// ADDITIONAL_COMPILE_FLAGS: -Wthread-safety

#include <mutex>

std::mutex m1;
std::mutex m2;
int data1 __attribute__((guarded_by(m1)));
int data2 __attribute__((guarded_by(m2)));

void no_lock_write() {
  data1++; // expected-warning {{writing variable 'data1' requires holding mutex 'm1' exclusively}}
}

void no_lock_write2() {
  std::scoped_lock<> lock;
  data1++; // expected-warning {{writing variable 'data1' requires holding mutex 'm1' exclusively}}
}

void wrong_lock_write() {
  std::scoped_lock<std::mutex> lock(m1);
  data2++; // expected-warning {{writing variable 'data2' requires holding mutex 'm2' exclusively}}
}

void adopt_without_lock() {
  // expected-warning@+1 {{calling function 'scoped_lock' requires holding mutex 'm1' exclusively}}
  std::scoped_lock<std::mutex> lock(std::adopt_lock, m1);
  data1++; // expected-warning {{writing variable 'data1' requires holding mutex 'm1' exclusively}}
}

void adopt_two_only_one_locked() {
  m1.lock();
  // expected-warning@+1 {{calling function 'scoped_lock' requires holding mutex 'm2' exclusively}}
  std::scoped_lock<std::mutex, std::mutex> lock(std::adopt_lock, m1, m2);
  data1++;
  data2++; // expected-warning {{writing variable 'data2' requires holding mutex 'm2' exclusively}}
}

void after_scoped_lock_scope() {
  {
    std::scoped_lock<std::mutex> lock(m1);
    data1++;
  }
  data1++; // expected-warning {{writing variable 'data1' requires holding mutex 'm1' exclusively}}
}
