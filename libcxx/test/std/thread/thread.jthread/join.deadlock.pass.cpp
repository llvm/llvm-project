//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Windows cannot detect the deadlock. Instead of throwing system_error,
// it would dead lock the test
// UNSUPPORTED: windows

// TSAN bug: https://llvm.org/PR66537
// UNSUPPORTED: tsan

// UNSUPPORTED: no-threads
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: c++03, c++11, c++14, c++17

// void join();

#include <atomic>
#include <cassert>
#include <chrono>
#include <concepts>
#include <functional>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**) {
  // resource_deadlock_would_occur - if deadlock is detected or get_id() == this_thread::get_id().
  {
    std::function<void()> f;
    std::atomic_bool start = false;
    std::atomic_bool done  = false;

    std::jthread jt = support::make_test_jthread([&] {
      start.wait(false);
      f();
      done = true;
      done.notify_all();
    });

    f = [&] {
      try {
        jt.join();
        assert(false);
      } catch (const std::system_error& err) {
        assert(err.code() == std::errc::resource_deadlock_would_occur);
      }
    };
    start = true;
    start.notify_all();
    done.wait(false);
  }

  return 0;
}
