//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// Check that functions are marked [[nodiscard]]

#include <chrono>
#include <barrier>
#include <future>
#include <latch>
#include <mutex>
#include <semaphore>
#include <thread>

#include "test_macros.h"

const auto timePoint = std::chrono::steady_clock::now();

void test() {
  { // [futures.async]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::async([]() {});
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::async(std::launch::any, []() {});
  }

  // std::scoped_lock
  {
#if TEST_STD_VER >= 17
    using M = std::mutex;
    M m0, m1, m2;
    // clang-format off
    std::scoped_lock<>{};                                   // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::scoped_lock<M>{m0};                                // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::scoped_lock<M, M>{m0, m1};                         // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::scoped_lock<M, M, M>{m0, m1, m2};                  // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}

    std::scoped_lock<>{std::adopt_lock};                    // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::scoped_lock<M>{std::adopt_lock, m0};               // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::scoped_lock<M, M>{std::adopt_lock, m0, m1};        // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::scoped_lock<M, M, M>{std::adopt_lock, m0, m1, m2}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    // clang-format on
#endif
  }

  // std::unique_lock
  {
    using M = std::timed_mutex; // necessary for the time_point and duration constructors
    M m;
    std::chrono::time_point<std::chrono::steady_clock> time_point;
    std::chrono::milliseconds duration;
    std::unique_lock<M> other;

    // clang-format off
    std::unique_lock<M>();                        // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    (std::unique_lock<M>)(m);                     // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>(m, std::defer_lock_t());  // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>(m, std::try_to_lock_t()); // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>(m, std::adopt_lock_t());  // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>(m, time_point);           // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>(m, duration);             // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>(std::move(other));        // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    // clang-format on
  }

  // std::lock_guard
  {
    std::mutex m;
    // clang-format off
    (std::lock_guard<std::mutex>)(m);                    // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::lock_guard<std::mutex>(m, std::adopt_lock_t()); // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    // clang-format on
  }

  // Threads
  {
    std::thread th;

    th.joinable();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    th.get_id();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    th.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    th.hardware_concurrency(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#if TEST_STD_VER >= 20
  {
    std::jthread jt;

    jt.joinable();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    jt.get_id();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    jt.native_handle();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    jt.get_stop_source(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    jt.get_stop_token();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    jt.hardware_concurrency(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif

  // Mutual exclusion

  { // <mutex>
    std::mutex m;

    m.try_lock();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::recursive_mutex m;

    m.try_lock();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::timed_mutex m;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.try_lock();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.try_lock_for(std::chrono::nanoseconds(82));
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.try_lock_until(timePoint);
  }
  {
    std::recursive_timed_mutex m;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.try_lock();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.try_lock_for(std::chrono::nanoseconds(82));
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.try_lock_until(timePoint);
  }
  {
    std::mutex m1;
    std::mutex m2;
    std::mutex m3;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::try_lock(m1, m2);
#if TEST_STD_VER >= 11
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::try_lock(m1, m2, m3);
#endif
  }

  // Condition variables

  { // <condition_variable>
    std::condition_variable cv;

    cv.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

#if TEST_STD_VER >= 20

  // Semaphores

  { // <semaphore>
    std::counting_semaphore<> cs{0};

    cs.max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    cs.try_acquire_for(std::chrono::nanoseconds{82});
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    cs.try_acquire();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    cs.try_acquire_until(timePoint);

    std::binary_semaphore bs{0};

    bs.max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    bs.try_acquire_for(std::chrono::nanoseconds{82});
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    bs.try_acquire();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    bs.try_acquire_until(timePoint);
  }

  // Latches and barriers

  { // <barrier>
    std::barrier<> b{94};

    b.max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  { // <latch>
    std::latch l{94};

    l.max();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    l.try_wait(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

#endif
}
