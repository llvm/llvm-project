//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr void pop_back();
// constexpr iterator iterator erase(const_iterator position);
// constexpr void clear() noexcept;
// constexpr ~inplace_vector();

#include <cassert>
#include <inplace_vector>
#include <type_traits>

#include "../common.h"
#include "test_macros.h"

struct Tracker {
  int* alive_;

  constexpr explicit Tracker(int& alive) : alive_(&alive) { ++*alive_; }
  constexpr Tracker(const Tracker& other) : alive_(other.alive_) { ++*alive_; }
  constexpr Tracker& operator=(const Tracker&) = default;
  constexpr ~Tracker() { --*alive_; }
};

struct NonTrivialDtor {
  ~NonTrivialDtor() {}
};

static_assert(std::is_nothrow_destructible_v<std::inplace_vector<int, 0> >);
static_assert(std::is_nothrow_destructible_v<std::inplace_vector<int, 8> >);
static_assert(std::is_nothrow_destructible_v<std::inplace_vector<NonTrivialDtor, 0> >);
static_assert(std::is_nothrow_destructible_v<std::inplace_vector<NonTrivialDtor, 8> >);

constexpr bool test() {
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    int alive = 0;
    {
      std::inplace_vector<Tracker, 8> c;
      c.emplace_back(alive);
      c.emplace_back(alive);
      c.emplace_back(alive);
      assert(alive == 3);

      c.pop_back();
      assert(alive == 2);

      c.erase(c.begin());
      assert(alive == 1);

      c.clear();
      assert(alive == 0);
    }
    assert(alive == 0);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
