//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <condition_variable>

// class condition_variable;

// void notify_one();

#include <condition_variable>
#include <atomic>
#include <mutex>
#include <thread>
#include <cassert>

#include "test_macros.h"


std::condition_variable cv;
std::mutex mut;

std::atomic_int test1(0);
std::atomic_int test2(0);
std::atomic_int ready(2);
std::atomic_int which(0);

void f1()
{
  --ready;
  std::unique_lock<std::mutex> lk(mut);
  assert(test1 == 0);
  while (test1 == 0)
    cv.wait(lk);
  which = 1;
  assert(test1 == 1);
  test1 = 2;
}

void f2()
{
  --ready;
  std::unique_lock<std::mutex> lk(mut);
  assert(test2 == 0);
  while (test2 == 0)
    cv.wait(lk);
  which = 2;
  assert(test2 == 1);
  test2 = 2;
}

int main(int, char**)
{
  std::thread t1(f1);
  std::thread t2(f2);
  while (ready > 0)
    std::this_thread::yield();
  // In case the threads were preempted right after the atomic decrement but
  // before cv.wait(), we yield one more time.
  std::this_thread::yield();
  {
    std::unique_lock<std::mutex>lk(mut);
    test1 = 1;
    test2 = 1;
    ready = 1;
  }
  cv.notify_one();
  {
    while (which == 0)
      std::this_thread::yield();
    std::unique_lock<std::mutex>lk(mut);
  }
  if (test1 == 2) {
    assert(test2 == 1);
    t1.join();
    test1 = 0;
  } else {
    assert(test1 == 1);
    assert(test2 == 2);
    t2.join();
    test2 = 0;
  }
  which = 0;
  cv.notify_one();
  {
    while (which == 0)
      std::this_thread::yield();
    std::unique_lock<std::mutex>lk(mut);
  }
  if (test1 == 2) {
    assert(test2 == 0);
    t1.join();
    test1 = 0;
  } else {
    assert(test1 == 0);
    assert(test2 == 2);
    t2.join();
    test2 = 0;
  }

  return 0;
}
