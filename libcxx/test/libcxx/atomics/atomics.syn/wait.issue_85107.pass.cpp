//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-threads

// This bug was first fixed in LLVM 19. This can't be XFAIL because
// the test is a no-op (and would XPASS) on some targets.
// UNSUPPORTED: using-built-library-before-llvm-19

// XFAIL: availability-synchronization_library-missing

// This is a regression test for https://llvm.org/PR85107, which describes how we were using UL_COMPARE_AND_WAIT instead
// of UL_COMPARE_AND_WAIT64 in the implementation of atomic::wait, leading to potential infinite hangs.

#include <atomic>
#include <cassert>
#include <chrono>

#include "make_test_thread.h"

int main(int, char**) {
  if constexpr (sizeof(std::__cxx_contention_t) == 8 && sizeof(long) > 4) {
    std::atomic<bool> done = false;
    auto const timeout     = std::chrono::system_clock::now() + std::chrono::seconds(600); // fail after 10 minutes

    auto timeout_thread = support::make_test_thread([&] {
      while (!done) {
        assert(std::chrono::system_clock::now() < timeout); // make sure we don't hang forever
      }
    });

    // https://llvm.org/PR85107
    // [libc++] atomic_wait uses UL_COMPARE_AND_WAIT when it should use UL_COMPARE_AND_WAIT64 on Darwin
    constexpr std::__cxx_contention_t old_val = 0;
    constexpr std::__cxx_contention_t new_val = old_val + (1ll << 32);
    std::__cxx_atomic_contention_t ct(new_val);

    // This would hang forever if the bug is present, but the test will fail in a bounded amount of
    // time due to the timeout above.
    std::__libcpp_atomic_wait(&ct, old_val);

    done = true;
    timeout_thread.join();
  }

  return 0;
}
