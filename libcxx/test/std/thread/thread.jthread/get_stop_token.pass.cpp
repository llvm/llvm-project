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

// [[nodiscard]] stop_token get_stop_token() const noexcept;

#include <cassert>
#include <concepts>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>

#include "make_test_thread.h"
#include "test_macros.h"

static_assert(noexcept(std::declval<const std::jthread&>().get_stop_token()));

int main(int, char**) {
  // Represents a thread
  {
    std::jthread jt                                 = support::make_test_jthread([] {});
    auto ss                                         = jt.get_stop_source();
    std::same_as<std::stop_token> decltype(auto) st = std::as_const(jt).get_stop_token();

    assert(st.stop_possible());
    assert(!st.stop_requested());
    ss.request_stop();
    assert(st.stop_requested());
  }

  // Does not represent a thread
  {
    const std::jthread jt{};
    std::same_as<std::stop_token> decltype(auto) st = jt.get_stop_token();

    assert(!st.stop_possible());
  }

  return 0;
}
