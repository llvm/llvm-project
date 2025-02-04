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

// [[nodiscard]] bool operator==(const stop_token& lhs, const stop_token& rhs) noexcept;
// Returns: true if lhs and rhs have ownership of the same stop state or if both lhs and rhs do not have ownership of a stop state; otherwise false.

// synthesized operator != also tested.

#include <cassert>
#include <concepts>
#include <stop_token>
#include <type_traits>

#include "test_macros.h"

// LWG 3254 is related.
template <class T>
concept IsNoThrowEqualityComparable = requires(const T& t1, const T& t2) {
  { t1 == t2 } noexcept;
};

template <class T>
concept IsNoThrowInequalityComparable = requires(const T& t1, const T& t2) {
  { t1 != t2 } noexcept;
};

static_assert(IsNoThrowEqualityComparable<std::stop_token>);
static_assert(IsNoThrowInequalityComparable<std::stop_token>);

int main(int, char**) {
  // both no state
  {
    const std::stop_token st1;
    const std::stop_token st2;
    assert(st1 == st2);
    assert(!(st1 != st2));
  }

  // only one has no state
  {
    std::stop_source ss;
    const std::stop_token st1;
    const auto st2 = ss.get_token();
    assert(!(st1 == st2));
    assert(st1 != st2);
  }

  // both has states. same source
  {
    std::stop_source ss;
    const auto st1 = ss.get_token();
    const auto st2 = ss.get_token();
    assert(st1 == st2);
    assert(!(st1 != st2));
  }

  // both has states. different sources with same states
  {
    std::stop_source ss1;
    auto ss2 = ss1;
    const auto st1 = ss1.get_token();
    const auto st2 = ss2.get_token();
    assert(st1 == st2);
    assert(!(st1 != st2));
  }

  // both has states. different sources with different states
  {
    std::stop_source ss1;
    std::stop_source ss2;
    const auto st1 = ss1.get_token();
    const auto st2 = ss2.get_token();
    assert(!(st1 == st2));
    assert(st1 != st2);
  }

  return 0;
}
