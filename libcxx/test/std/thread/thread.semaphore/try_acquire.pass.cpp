//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// XFAIL: availability-synchronization_library-missing

// <semaphore>

#include <cassert>
#include <semaphore>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**)
{
  std::counting_semaphore<> s(1);

  assert(s.try_acquire());
  assert(!s.try_acquire());
  s.release();
  assert(s.try_acquire());
  assert(!s.try_acquire());
  s.release(2);
  std::thread t = support::make_test_thread([&](){
    assert(s.try_acquire());
  });
  t.join();
  assert(s.try_acquire());
  assert(!s.try_acquire());

  return 0;
}
