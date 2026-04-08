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

// [[nodiscard]] bool operator==(const stop_source& lhs, const stop_source& rhs) noexcept;
// Returns: true if lhs and rhs have ownership of the same stop state or if both lhs and rhs do not have ownership of a stop state; otherwise false.

#include <cassert>
#include <concepts>
#include <stop_token>
#include <type_traits>

#include "test_macros.h"

template <class T>
concept IsNoThrowEqualityComparable = requires(const T& t1, const T& t2) {
  { t1 == t2 } noexcept;
};

static_assert(IsNoThrowEqualityComparable<std::stop_source>);

int main(int, char**) {
  // both no state
  {
    const std::stop_source ss1(std::nostopstate);
    const std::stop_source ss2(std::nostopstate);
    assert(ss1 == ss2);
    assert(!(ss1 != ss2));
  }

  // only one has no state
  {
    const std::stop_source ss1(std::nostopstate);
    const std::stop_source ss2;
    assert(!(ss1 == ss2));
    assert(ss1 != ss2);
  }

  // both has states. same state
  {
    const std::stop_source ss1;
    const std::stop_source ss2(ss1);
    assert(ss1 == ss2);
    assert(!(ss1 != ss2));
  }

  // both has states. different states
  {
    const std::stop_source ss1;
    const std::stop_source ss2;
    assert(!(ss1 == ss2));
    assert(ss1 != ss2);
  }

  return 0;
}
