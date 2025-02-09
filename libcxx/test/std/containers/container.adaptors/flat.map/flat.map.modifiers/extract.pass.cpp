//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// containers extract() &&;

#include <algorithm>
#include <concepts>
#include <deque>
#include <flat_map>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class T>
concept CanExtract = requires(T&& t) { std::forward<T>(t).extract(); };

static_assert(CanExtract<std::flat_map<int, int>&&>);
static_assert(!CanExtract<std::flat_map<int, int>&>);
static_assert(!CanExtract<std::flat_map<int, int> const&>);
static_assert(!CanExtract<std::flat_map<int, int> const&&>);

template <class KeyContainer, class ValueContainer>
void test() {
  using M = std::flat_map<int, int, std::less<int>, KeyContainer, ValueContainer>;
  M m     = M({1, 2, 3}, {4, 5, 6});

  std::same_as<typename M::containers> auto containers = std::move(m).extract();

  auto expected_keys   = {1, 2, 3};
  auto expected_values = {4, 5, 6};
  assert(std::ranges::equal(containers.keys, expected_keys));
  assert(std::ranges::equal(containers.values, expected_values));
  check_invariant(m);
  LIBCPP_ASSERT(m.empty());
  LIBCPP_ASSERT(m.keys().size() == 0);
  LIBCPP_ASSERT(m.values().size() == 0);
}

int main(int, char**) {
  test<std::vector<int>, std::vector<int>>();
  test<std::deque<int>, std::vector<int>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>();
  {
    // extracted object maintains invariant if one of underlying container does not clear after move
    using M = std::flat_map<int, int, std::less<>, std::vector<int>, CopyOnlyVector<int>>;
    M m     = M({1, 2, 3}, {1, 2, 3});
    std::same_as<M::containers> auto containers = std::move(m).extract();
    assert(containers.keys.size() == 3);
    assert(containers.values.size() == 3);
    check_invariant(m);
    LIBCPP_ASSERT(m.empty());
    LIBCPP_ASSERT(m.keys().size() == 0);
    LIBCPP_ASSERT(m.values().size() == 0);
  }

  {
#ifndef TEST_HAS_NO_EXCEPTIONS
    using KeyContainer   = std::vector<int>;
    using ValueContainer = ThrowOnMoveContainer<int>;
    using M              = std::flat_map<int, int, std::ranges::less, KeyContainer, ValueContainer>;

    M m;
    m.emplace(1, 1);
    m.emplace(2, 2);
    try {
      auto c = std::move(m).extract();
      assert(false);
    } catch (int) {
      check_invariant(m);
      // In libc++, we try to erase the key after value emplacement failure.
      // and after erasure failure, we clear the flat_map
      LIBCPP_ASSERT(m.size() == 0);
    }
#endif
  }
  return 0;
}
