//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// Make sure that we warn on unused variables of libc++ classes which behave like value types.
// ADDITIONAL_COMPILE_FLAGS: -Wunused-variable

#include <barrier>
#include <condition_variable>
#include <latch>
#include <mutex>
#include <semaphore>
#include <shared_mutex>

#include "test_macros.h"

void synchronization(std::mutex& m) {
  // <mutex>
  std::mutex a;                      // expected-warning {{unused variable}}
  std::once_flag b;                  // expected-warning {{unused variable}}
  std::recursive_mutex c;            // expected-warning {{unused variable}}
  std::recursive_timed_mutex d;      // expected-warning {{unused variable}}
  std::timed_mutex e;                // expected-warning {{unused variable}}
  std::unique_lock<std::mutex> f;    // This is https://llvm.org/PR222655
  std::unique_lock<std::mutex> g(m); // Shouldn't be diagnosed
#if TEST_STD_VER >= 17
  std::unique_lock<std::mutex> h(m, std::defer_lock); // This is https://llvm.org/PR222655
#endif

  // <condition_variable>
  std::condition_variable i;     // expected-warning {{unused variable}}
  std::condition_variable_any j; // expected-warning {{unused variable}}

#if TEST_STD_VER >= 20
  // <semaphore>
  std::counting_semaphore<> k(1); // expected-warning {{unused variable}}
#endif

  // <shared_mutex>
#if TEST_STD_VER >= 17
  std::shared_mutex l; // expected-warning {{unused variable}}
#endif
#if TEST_STD_VER >= 14
  std::shared_timed_mutex n; // expected-warning {{unused variable}}
  std::shared_timed_mutex n2;
  std::shared_lock<std::shared_timed_mutex> o;                      // This is https://llvm.org/PR222655
  std::shared_lock<std::shared_timed_mutex> p(n2);                  // Shouldn't be diagnosed
  std::shared_lock<std::shared_timed_mutex> q(n2, std::defer_lock); // This is https://llvm.org/PR222655
#endif

#if TEST_STD_VER >= 20
  // <barrier>
  std::barrier<> r(1); // expected-warning {{unused variable}}

  // <latch>
  std::latch s(1); // expected-warning {{unused variable}}
#endif
}
