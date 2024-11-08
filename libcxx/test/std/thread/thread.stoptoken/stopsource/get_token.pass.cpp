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

// [[nodiscard]] stop_token get_token() const noexcept;

#include <cassert>
#include <concepts>
#include <stop_token>
#include <type_traits>

#include "test_macros.h"

template <class T>
concept IsGetTokenNoexcept = requires(const T& t) {
  { t.get_token() } noexcept;
};

static_assert(IsGetTokenNoexcept<std::stop_source>);

int main(int, char**) {
  // no state
  {
    std::stop_source ss{std::nostopstate};
    std::same_as<std::stop_token> decltype(auto) st = ss.get_token();
    assert(!st.stop_possible());
    assert(!st.stop_requested());
  }

  // with state
  {
    std::stop_source ss;
    std::same_as<std::stop_token> decltype(auto) st = ss.get_token();
    assert(st.stop_possible());
    assert(!st.stop_requested());

    ss.request_stop();
    assert(st.stop_requested());
  }

  return 0;
}
