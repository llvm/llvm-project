//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-experimental-stop_token
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// [[nodiscard]] bool joinable() const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <thread>
#include <type_traits>

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
    const std::jthread jt{[] {}};
    std::same_as<bool> decltype(auto) result = jt.joinable();
    assert(result);
  }

  // Non-default constructed
  // the thread of execution has not finished
  {
    std::atomic_bool done = false;
    const std::jthread jt{[&done] { done.wait(false); }};
    std::same_as<bool> decltype(auto) result = jt.joinable();
    done                                     = true;
    done.notify_all();
    assert(result);
  }

  return 0;
}
