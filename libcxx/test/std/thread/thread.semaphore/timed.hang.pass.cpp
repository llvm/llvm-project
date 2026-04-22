//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <semaphore>

// This is a regression test for a bug in semaphore::try_acquire_for
// where it can wait indefinitely
// https://github.com/llvm/llvm-project/issues/180334

#include <semaphore>
#include <thread>
#include <chrono>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

void test() {
  auto const start = std::chrono::steady_clock::now();
  std::counting_semaphore<> s(0);

  assert(!s.try_acquire_for(std::chrono::nanoseconds(1)));
  assert(!s.try_acquire_for(std::chrono::microseconds(1)));
  assert(!s.try_acquire_for(std::chrono::milliseconds(1)));
  assert(!s.try_acquire_for(std::chrono::milliseconds(100)));

  auto const end = std::chrono::steady_clock::now();
  assert(end - start < std::chrono::seconds(10));
}

int main(int, char**) {
  for (auto i = 0; i < 10; ++i) {
    test();
  }

  return 0;
}
