#ifndef TEST_STD_THREAD_THREAD_MUTEX_THREAD_LOCK_THREAD_LOCK_GUARD_TYPES_H
#define TEST_STD_THREAD_THREAD_MUTEX_THREAD_LOCK_THREAD_LOCK_GUARD_TYPES_H

#include <cassert>

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

  MyMutex(MyMutex const&)            = delete;
  MyMutex& operator=(MyMutex const&) = delete;
};

#endif // TEST_STD_THREAD_THREAD_MUTEX_THREAD_LOCK_THREAD_LOCK_GUARD_TYPES_H
