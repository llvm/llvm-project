//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// This test occasionally fails on Android.
// UNSUPPORTED: LIBCXX-ANDROID-FIXME

// <condition_variable>

// class condition_variable;

// void notify_one();


// NOTE: `notify_one` is just a wrapper around pthread_cond_signal, but
// POSIX does not guarantee that one and only one thread will be woken:
//
// https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_cond_signal.html
//
// Quote:
//     Multiple Awakenings by Condition Signal
//     On a multi-processor, it may be impossible for an implementation of
//     pthread_cond_signal() to avoid the unblocking of more than one thread
//     blocked on a condition variable. For example...



// NOTE: In previous versions of this test, `notify_one` was called WITHOUT
// holding the lock but POSIX says (in the aforementioned URL) that:
//     ...if predictable scheduling behavior is required, then that mutex shall
//     be locked by the thread calling pthread_cond_broadcast() or
//     pthread_cond_signal().


#include <condition_variable>
#include <atomic>
#include <mutex>
#include <thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"


std::condition_variable cv;
std::mutex mut;

std::atomic_int test1(0);
std::atomic_int test2(0);
std::atomic_int ready(2);
std::atomic_int which(0);

void f1()
{
  std::unique_lock<std::mutex> lk(mut);
  assert(test1 == 0);
  --ready;
  while (test1 == 0)
    cv.wait(lk);
  which = 1;
  assert(test1 == 1);
  test1 = 2;
}

void f2()
{
  std::unique_lock<std::mutex> lk(mut);
  assert(test2 == 0);
  --ready;
  while (test2 == 0)
    cv.wait(lk);
  which = 2;
  assert(test2 == 1);
  test2 = 2;
}

int main(int, char**)
{
  std::thread t1 = support::make_test_thread(f1);
  std::thread t2 = support::make_test_thread(f2);
  {
    while (ready > 0)
      std::this_thread::yield();
    // At this point:
    // 1) Both f1 and f2 have entered their condition variable wait.
    // 2) Either f1 or f2 has the mutex locked and is about to wait.
    std::unique_lock<std::mutex> lk(mut);
    test1 = 1;
    test2 = 1;
    ready = 1;
    cv.notify_one();
  }
  {
    while (which == 0)
      std::this_thread::yield();
    std::unique_lock<std::mutex> lk(mut);
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
  }
  {
    while (which == 0)
      std::this_thread::yield();
    std::unique_lock<std::mutex> lk(mut);
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
  }

  return 0;
}
