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
int data3;

static void no_mutex() {
  std::scoped_lock<> lock;
  data3++;
}

static void one_mutex() {
  std::scoped_lock<std::mutex> lock(m1);
  data1++;
}

static void two_mutexes() {
  std::scoped_lock<std::mutex, std::mutex> lock(m1, m2);
  data1++;
  data2++;
}

static void adopt_none() {
  std::scoped_lock<> lock(std::adopt_lock);
  data3++;
}

static void adopt_one() {
  m1.lock();
  std::scoped_lock<std::mutex> lock(std::adopt_lock, m1);
  data1++;
}

static void adopt_two() {
  m1.lock();
  m2.lock();
  std::scoped_lock<std::mutex, std::mutex> lock(std::adopt_lock, m1, m2);
  data1++;
  data2++;
}

int main(int, char**) {
  no_mutex();
  one_mutex();
  two_mutexes();
  adopt_none();
  adopt_one();
  adopt_two();
  return 0;
}
