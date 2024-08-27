//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_THREAD_THREAD_MUTEX_THREAD_LOCK_THREAD_LOCK_GUARD_TYPES_H
#define TEST_STD_THREAD_THREAD_MUTEX_THREAD_LOCK_THREAD_LOCK_GUARD_TYPES_H

#include <cassert>
#include <chrono>

struct MyMutex {
  bool locked = false;

  MyMutex() = default;
  ~MyMutex() { assert(!locked); }

  void lock() {
    assert(!locked);
    locked = true;
  }

  void unlock() {
    assert(locked);
    locked = false;
  }

  bool try_lock() {
    if (locked)
      return false;
    lock();
    return true;
  }

  template <class Rep, class Period>
  bool try_lock_for(const std::chrono::duration<Rep, Period>& rel_time) {
    using ms = std::chrono::milliseconds;
    assert(rel_time == ms(5));
    if (locked)
      return false;
    lock();
    return true;
  }

  MyMutex(MyMutex const&)            = delete;
  MyMutex& operator=(MyMutex const&) = delete;
};

struct MyTimedMutex {
  using ms = std::chrono::milliseconds;

  bool try_lock_called       = false;
  bool try_lock_for_called   = false;
  bool try_lock_until_called = false;

  bool try_lock() {
    try_lock_called = !try_lock_called;
    return try_lock_called;
  }

  template <class Rep, class Period>
  bool try_lock_for(const std::chrono::duration<Rep, Period>& rel_time) {
    assert(rel_time == ms(5));
    try_lock_for_called = !try_lock_for_called;
    return try_lock_for_called;
  }

  template <class Clock, class Duration>
  bool try_lock_until(const std::chrono::time_point<Clock, Duration>& abs_time) {
    assert(Clock::now() - abs_time < ms(5));
    try_lock_until_called = !try_lock_until_called;
    return try_lock_until_called;
  }

  void unlock() {}
};

struct MyCountingMutex {
  static int lock_count;
  static int unlock_count;
  void lock() { ++lock_count; }
  void unlock() { ++unlock_count; }
};

#endif // TEST_STD_THREAD_THREAD_MUTEX_THREAD_LOCK_THREAD_LOCK_GUARD_TYPES_H
