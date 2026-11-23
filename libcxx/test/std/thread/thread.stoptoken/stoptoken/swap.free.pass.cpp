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

// friend void swap(stop_token& x, stop_token& y) noexcept;

#include <cassert>
#include <concepts>
#include <stop_token>
#include <type_traits>

#include "test_macros.h"

template <class T>
concept IsNoThrowFreeSwappable = requires(T& t) {
  { swap(t, t) } noexcept;
};

static_assert(IsNoThrowFreeSwappable<std::stop_token>);

int main(int, char**) {
  {
    std::stop_token st1;

    std::stop_source source;
    auto st2 = source.get_token();

    assert(st1 != st2);

    source.request_stop();

    assert(!st1.stop_requested());
    assert(st2.stop_requested());

    swap(st1, st2);

    assert(st1 != st2);
    assert(st1.stop_requested());
    assert(!st2.stop_requested());
  }

  return 0;
}
