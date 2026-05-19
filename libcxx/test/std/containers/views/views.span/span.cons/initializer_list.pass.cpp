//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

#include <any>
#include <cassert>
#include <cstddef>
#include <span>

#include "test_convertible.h"
#include "test_macros.h"

struct Sink {
  constexpr Sink() = default;
  constexpr Sink(Sink*) {}
};

constexpr std::size_t count(std::span<const Sink> sp) { return sp.size(); }

template <std::size_t N>
constexpr std::size_t count_n(std::span<const Sink, N> sp) {
  return sp.size();
}

constexpr bool test() {
  Sink a[10];

  assert(count({a}) == 10);
  assert(count({a, a + 10}) == 10);
  assert(count_n<10>({a}) == 10);

  return true;
}

// Test P2447R4 "Annex C examples"
// P2447R4 was reverted by P4144R1, so we only test the "old" behavior.

constexpr int three(std::span<void* const> sp) { return static_cast<int>(sp.size()); }

constexpr int four(std::span<const std::any> sp) { return static_cast<int>(sp.size()); }

bool test_P2447R4_annex_c_examples() {
  {
    void* a[10];
    assert(three({a, 0}) == 0);
  }
  {
    std::any a[10];
    assert(four({a, a + 10}) == 10);
  }

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  assert(test_P2447R4_annex_c_examples());

  return 0;
}
