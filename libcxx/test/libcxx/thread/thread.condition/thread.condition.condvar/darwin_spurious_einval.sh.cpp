//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{.+}}-apple-{{.+}}
// REQUIRES: libcpp-has-thread-api-pthread
// REQUIRES: has-unix-headers
// UNSUPPORTED: no-threads, no-exceptions, c++03
// XFAIL: using-built-library-before-llvm-24

// <condition_variable>

// Test that a kernel failure of the wait, which Darwin reports as EINVAL with
// errno set to EBUSY or EINTR, is a spurious wakeup, and that other EINVAL
// results still terminate. The failures are injected by interposing
// pthread_cond_wait and pthread_cond_timedwait from a dylib.

// RUN: %{cxx} %s %{flags} %{compile_flags} %{link_flags} -DINTERPOSER -shared \
// RUN:   -Wl,-install_name,@executable_path/interposer.dylib -o %{temp}/interposer.dylib
// RUN: %{cxx} %s %{flags} %{compile_flags} %{link_flags} %{temp}/interposer.dylib -DMAIN -o %t.exe
// RUN: %{exec} %t.exe

#include <cerrno>
#include <pthread.h>

// The next `failed_waits` calls return EINVAL and set errno to `failed_wait_errno`.
// The next `failed_wakeups` calls do the same after waking up.
extern "C" int failed_waits;
extern "C" int failed_wakeups;
extern "C" int failed_wait_errno;

#if defined(INTERPOSER)

int failed_waits      = 0;
int failed_wakeups    = 0;
int failed_wait_errno = 0;

static bool fail(int& count) {
  if (count == 0)
    return false;
  --count;
  errno = failed_wait_errno;
  return true;
}

static int interposed_wait(pthread_cond_t* cv, pthread_mutex_t* m) {
  if (fail(failed_waits))
    return EINVAL;
  int ec = pthread_cond_wait(cv, m);
  return ec == 0 && fail(failed_wakeups) ? EINVAL : ec;
}

static int interposed_timedwait(pthread_cond_t* cv, pthread_mutex_t* m, const timespec* ts) {
  if (fail(failed_waits))
    return EINVAL;
  int ec = pthread_cond_timedwait(cv, m, ts);
  return ec == 0 && fail(failed_wakeups) ? EINVAL : ec;
}

struct Interpose {
  const void* replacement;
  const void* original;
};

__attribute__((used, section("__DATA,__interpose"))) static const Interpose interposers[] = {
    {reinterpret_cast<const void*>(&interposed_wait), reinterpret_cast<const void*>(&pthread_cond_wait)},
    {reinterpret_cast<const void*>(&interposed_timedwait), reinterpret_cast<const void*>(&pthread_cond_timedwait)},
};

#elif defined(MAIN)

#  include <atomic>
#  include <cassert>
#  include <chrono>
#  include <condition_variable>
#  include <mutex>
#  include <new>
#  include <thread>

#  include "check_assertion.h"
#  include "test_macros.h"

int main(int, char**) {
  std::mutex m;
  std::condition_variable cv;
  std::unique_lock<std::mutex> lk(m);

  // Otherwise the waits below never return.
  failed_waits      = 1;
  failed_wait_errno = EBUSY;
  assert(pthread_cond_wait(cv.native_handle(), m.native_handle()) == EINVAL);
  assert(failed_waits == 0);

  for (int e : {EBUSY, EINTR}) {
    failed_waits      = 1;
    failed_wait_errno = e;
    errno             = 42;
    cv.wait(lk);
    assert(failed_waits == 0);
    assert(lk.owns_lock());
    assert(errno == 42);
  }

  {
    int calls         = 0;
    failed_waits      = 1;
    failed_wait_errno = EBUSY;
    cv.wait(lk, [&] { return ++calls == 2; });
    assert(calls == 2);
    assert(lk.owns_lock());
  }

  {
    auto timeout      = std::chrono::milliseconds(50);
    auto start        = std::chrono::steady_clock::now();
    failed_waits      = 1;
    failed_wait_errno = EINTR;
    errno             = 42;
    bool result       = cv.wait_for(lk, timeout, [] { return false; });
    assert(!result);
    assert(std::chrono::steady_clock::now() - start >= timeout);
    assert(lk.owns_lock());
    assert(errno == 42);
  }

  // Invalid arguments are reported before waiting, without setting errno.
  failed_waits      = 1;
  failed_wait_errno = 0;
  EXPECT_STD_TERMINATE([&] { cv.wait(lk); });
  EXPECT_STD_TERMINATE([&] { cv.wait_for(lk, std::chrono::seconds(1)); });
  failed_waits = 0;

  // The condition variable can be destroyed once its waiter is notified.
  lk.unlock();
  for (bool timed : {false, true}) {
    alignas(std::condition_variable) unsigned char buffer[sizeof(std::condition_variable)];
    std::condition_variable* destroyed = new (buffer) std::condition_variable;
    std::atomic<bool> waiting(false);
    std::thread waiter([&] {
      std::unique_lock<std::mutex> waiter_lk(m);
      waiting = true;
      if (timed)
        destroyed->wait_for(waiter_lk, std::chrono::seconds(60));
      else
        destroyed->wait(waiter_lk);
    });
    while (!waiting)
      std::this_thread::yield();
    {
      std::lock_guard<std::mutex> guard(m);
      failed_wakeups    = 1;
      failed_wait_errno = EBUSY;
      destroyed->notify_one();
      destroyed->~condition_variable();
    }
    waiter.join();
  }

  return 0;
}

#endif
