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
// XFAIL: availability-synchronization_library-missing

// [[nodiscard]] bool request_stop() noexcept;

#include <cassert>
#include <concepts>
#include <stop_token>
#include <thread>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

static_assert(noexcept(std::declval<std::jthread&>().request_stop()));

int main(int, char**) {
  // Represents a thread
  {
    std::jthread jt = support::make_test_jthread([] {});
    auto st         = jt.get_stop_token();
    assert(!st.stop_requested());
    std::same_as<bool> decltype(auto) result = jt.request_stop();
    assert(result);
    assert(st.stop_requested());
  }

  // Does not represent a thread
  {
    std::jthread jt{};
    std::same_as<bool> decltype(auto) result = jt.request_stop();
    assert(!result);
  }

  return 0;
}
