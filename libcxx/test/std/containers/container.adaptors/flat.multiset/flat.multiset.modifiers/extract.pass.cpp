//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// container_type extract() &&;

#include <algorithm>
#include <concepts>
#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class T>
concept CanExtract = requires(T&& t) { std::forward<T>(t).extract(); };

static_assert(CanExtract<std::flat_multiset<int>&&>);
static_assert(!CanExtract<std::flat_multiset<int>&>);
static_assert(!CanExtract<std::flat_multiset<int> const&>);
static_assert(!CanExtract<std::flat_multiset<int> const&&>);

template <class KeyContainer>
void test_one() {
  using M = std::flat_multiset<int, std::less<int>, KeyContainer>;
  {
    M m = M({1, 1, 3});

    std::same_as<KeyContainer> auto keys = std::move(m).extract();

    auto expected_keys = {1, 1, 3};
    assert(std::ranges::equal(keys, expected_keys));
    check_invariant(m);
    LIBCPP_ASSERT(m.empty());
  }
  {
    // was empty
    M m;
    assert(m.empty());
    auto keys = std::move(m).extract();
    assert(keys.empty());
    LIBCPP_ASSERT(m.empty());
  }
}

void test() {
  test_one<std::vector<int>>();
  test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();

  {
    // extracted object maintains invariant if the underlying container does not clear after move
    using M                                   = std::flat_multiset<int, std::less<>, CopyOnlyVector<int>>;
    M m                                       = M({1, 1, 3});
    std::same_as<M::container_type> auto keys = std::move(m).extract();
    assert(keys.size() == 3);
    check_invariant(m);
    LIBCPP_ASSERT(m.empty());
  }
}

void test_exception() {
  {
#ifndef TEST_HAS_NO_EXCEPTIONS
    using KeyContainer = ThrowOnMoveContainer<int>;
    using M            = std::flat_multiset<int, std::ranges::less, KeyContainer>;

    M m;
    m.emplace(1);
    m.emplace(2);
    try {
      auto c = std::move(m).extract();
      assert(false);
    } catch (int) {
      check_invariant(m);
      // In libc++, we try to erase the key after value emplacement failure.
      // and after erasure failure, we clear the flat_multiset
      LIBCPP_ASSERT(m.size() == 0);
    }
#endif
  }
}

int main(int, char**) {
  test();
  test_exception();

  return 0;
}
