//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// UNSUPPORTED: no-threads

// <condition_variable>

// class condition_variable_any;

// RUN: %{build}
// RUN: %{run} 1
// RUN: %{run} 2
// RUN: %{run} 3
// RUN: %{run} 4
// RUN: %{run} 5
// RUN: %{run} 6
// RUN: %{run} 7
// RUN: %{run} 8
// RUN: %{run} 9

// -----------------------------------------------------------------------------
// Overview
//   Check that std::terminate is called if wait(...) fails to meet its post
//   conditions. This can happen when reacquiring the mutex throws
//   an exception.
//
//  The following methods are tested within this file
//   1.  void wait(Lock& lock);
//   2.  void wait(Lock& lock, Pred);
//   3.  void wait_for(Lock& lock, Duration);
//   4.  void wait_for(Lock& lock, Duration, Pred);
//   5.  void wait_until(Lock& lock, TimePoint);
//   6.  void wait_until(Lock& lock, TimePoint, Pred);
//   7.  bool wait(Lock& lock, stop_token stoken, Predicate pred);
//   8.  bool wait_for(Lock& lock, stop_token stoken, Duration, Predicate pred);
//   9.  bool wait_until(Lock& lock, stop_token stoken, TimePoint, Predicate pred);
//
// Plan
//   1 Create a mutex type, 'ThrowingMutex', that throws when the lock is acquired
//     for the *second* time.
//
//   2 Replace the terminate handler with one that exits with a '0' exit code.
//
//   3 Create a 'condition_variable_any' object 'cv' and a 'ThrowingMutex'
//     object 'm' and lock 'm'.
//
//   4 Start a thread 'T2' that will notify 'cv' once 'm' has been unlocked.
//
//   5 From the main thread call the specified wait method on 'cv' with 'm'.
//     When 'T2' notifies 'cv' and the wait method attempts to re-lock
//    'm' an exception will be thrown from 'm.lock()'.
//
//   6 Check that control flow does not return from the wait method and that
//     terminate is called (If the program exits with a 0 exit code we know
//     that terminate has been called)


#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <string>
#include <stop_token>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

void my_terminate() {
  std::_Exit(0); // Use _Exit to prevent cleanup from taking place.
}

// The predicate used in the cv.wait calls.
bool pred = false;
bool pred_function() {
  return pred == true;
}

class ThrowingMutex
{
  std::atomic_bool locked;
  unsigned state = 0;
  ThrowingMutex(const ThrowingMutex&) = delete;
  ThrowingMutex& operator=(const ThrowingMutex&) = delete;
public:
  ThrowingMutex() {
    locked = false;
  }
  ~ThrowingMutex() = default;

  void lock() {
    locked = true;
    if (++state == 2) {
      assert(pred); // Check that we actually waited until we were signaled.
      throw 1;  // this throw should end up calling terminate()
    }
  }

  void unlock() { locked = false; }
  bool isLocked() const { return locked == true; }
};

ThrowingMutex mut;
std::condition_variable_any cv;

void signal_me() {
  while (mut.isLocked()) {} // wait until T1 releases mut inside the cv.wait call.
  pred = true;
  cv.notify_one();
}

typedef std::chrono::system_clock Clock;
typedef std::chrono::milliseconds MS;

int main(int argc, char **argv) {
  assert(argc == 2);
  int id = std::stoi(argv[1]);
  assert(id >= 1 && id <= 9);
  std::set_terminate(my_terminate); // set terminate after std::stoi because it can throw.
  MS wait(250);
  try {
    mut.lock();
    assert(pred == false);
    support::make_test_thread(signal_me).detach();
    switch (id) {
      case 1: cv.wait(mut); break;
      case 2: cv.wait(mut, pred_function); break;
      case 3: cv.wait_for(mut, wait); break;
      case 4: cv.wait_for(mut, wait, pred_function); break;
      case 5: cv.wait_until(mut, Clock::now() + wait); break;
      case 6: cv.wait_until(mut, Clock::now() + wait, pred_function); break;
#if TEST_STD_VER >= 20 && !(defined(_LIBCPP_VERSION) && !_LIBCPP_AVAILABILITY_HAS_SYNC)
      case 7: cv.wait(mut, std::stop_source{}.get_token(), pred_function); break;
      case 8: cv.wait_for(mut, std::stop_source{}.get_token(), wait, pred_function); break;
      case 9: cv.wait_until(mut, std::stop_source{}.get_token(), Clock::now() + wait, pred_function); break;
#else
      case 7:
      case 8:
      case 9:
        return 0;
#endif
      default: assert(false);
    }
  } catch (...) {}
  assert(false);

  return 0;
}
