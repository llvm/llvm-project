//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// void detach();

#include <atomic>
#include <cassert>
#include <chrono>
#include <concepts>
#include <functional>
#include <optional>
#include <system_error>
#include <thread>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**) {
  // Effects: The thread represented by *this continues execution without the calling thread blocking.
  {
    std::atomic_bool start{false};
    std::atomic_bool done{false};
    std::optional<std::jthread> jt = support::make_test_jthread([&start, &done] {
      start.wait(false);
      done = true;
    });

    // If it blocks, it will deadlock here
    jt->detach();

    jt.reset();

    // The other thread continues execution
    start = true;
    start.notify_all();
    while (!done) {
    }
  }

  // Postconditions: get_id() == id().
  {
    std::jthread jt = support::make_test_jthread([] {});
    assert(jt.get_id() != std::jthread::id());
    jt.detach();
    assert(jt.get_id() == std::jthread::id());
  }

#if !defined(TEST_HAS_NO_EXCEPTIONS)
  // Throws: system_error when an exception is required ([thread.req.exception]).
  // invalid_argument - if the thread is not joinable.
  {
    std::jthread jt;
    try {
      jt.detach();
      assert(false);
    } catch (const std::system_error& err) {
      assert(err.code() == std::errc::invalid_argument);
    }
  }
#endif

  std::this_thread::sleep_for(std::chrono::milliseconds{2});
  return 0;
}
