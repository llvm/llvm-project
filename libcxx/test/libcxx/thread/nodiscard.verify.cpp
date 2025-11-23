//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-threads

// Check that functions are marked [[nodiscard]]

#include <barrier>
#include <latch>
#include <mutex>
#include <semaphore>
#include <thread>

#include "test_macros.h"

void test() {
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

    m.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::recursive_mutex m;

    m.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  // Condition variables

  { // <condition_variable>
    std::condition_variable cv;

    cv.native_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

#if TEST_STD_VER >= 20

  // Semaphores

  { // <semaphor>
    std::counting_semaphore<> cs{0};

    cs.max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::binary_semaphore bs{0};

    bs.max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  // Latches and barriers

  { // <barrier>
    std::barrier<> b{94};

    b.max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  { // <latch>
    std::latch l{94};

    l.max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

#endif
}
