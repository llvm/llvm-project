//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// UNSUPPORTED: no-threads

// check that <mutex> functions are marked [[nodiscard]]

#include <mutex>
#include <chrono>
#include <utility>

#include "test_macros.h"

void test() {
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
    std::unique_lock<M>{};                      // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>{m};                     // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>{m, std::defer_lock};    // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>{m, std::try_to_lock};   // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>{m, std::adopt_lock};    // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>{m, time_point};         // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>{m, duration};           // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::unique_lock<M>(std::move(other));      // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    // clang-format on
  }

  // std::lock_guard
  {
    std::mutex m;
    // clang-format off
    std::lock_guard<std::mutex>{m};                  // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    std::lock_guard<std::mutex>{m, std::adopt_lock}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
    // clang-format on
  }
}
