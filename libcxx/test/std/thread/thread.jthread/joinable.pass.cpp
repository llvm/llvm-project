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

// [[nodiscard]] bool joinable() const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <thread>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

static_assert(noexcept(std::declval<const std::jthread&>().joinable()));

int main(int, char**) {
  // Default constructed
  {
    const std::jthread jt;
    std::same_as<bool> decltype(auto) result = jt.joinable();
    assert(!result);
  }

  // Non-default constructed
  {
    const std::jthread jt                    = support::make_test_jthread([] {});
    std::same_as<bool> decltype(auto) result = jt.joinable();
    assert(result);
  }

  // Non-default constructed
  // the thread of execution has not finished
  {
    std::atomic_bool done                    = false;
    const std::jthread jt                    = support::make_test_jthread([&done] { done.wait(false); });
    std::same_as<bool> decltype(auto) result = jt.joinable();
    done                                     = true;
    done.notify_all();
    assert(result);
  }

  return 0;
}
