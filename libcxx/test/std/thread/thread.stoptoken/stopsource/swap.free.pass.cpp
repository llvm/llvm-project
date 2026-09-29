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

// void swap(stop_source& rhs) noexcept;

#include <cassert>
#include <concepts>
#include <stop_token>
#include <type_traits>

#include "test_macros.h"

template <class T>
concept IsNoThrowFreeSwappable = requires(T& t) {
  { swap(t, t) } noexcept;
};

static_assert(IsNoThrowFreeSwappable<std::stop_source>);

int main(int, char**) {
  {
    std::stop_source ss1;
    std::stop_source ss2;

    assert(ss1 != ss2);

    ss2.request_stop();

    assert(!ss1.stop_requested());
    assert(ss2.stop_requested());

    swap(ss1, ss2);

    assert(ss1 != ss2);
    assert(ss1.stop_requested());
    assert(!ss2.stop_requested());
  }

  return 0;
}
