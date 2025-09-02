//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template <class... Args>
//  iterator emplace(Args&&... args);

#include <flat_set>
#include <cassert>
#include <deque>
#include <tuple>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "../../../Emplaceable.h"
#include "DefaultOnly.h"
#include "min_allocator.h"

template <class KeyContainer>
void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_multiset<Key, std::less<Key>, KeyContainer>;
  using R   = typename M::iterator;
  {
    // was empty
    M m;
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r == m.begin());
    assert(m.size() == 1);
    assert(*r == 2);
  }
  {
    // key does not exist and inserted at the begin
    M m                              = {3, 3, 3, 7};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r == m.begin());
    assert(m.size() == 5);
    assert(*r == 2);
  }
  {
    // key does not exist and inserted in the middle
    M m                              = {1, 1, 3, 4};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(*r == 2);
  }
  {
    // key does not exist and inserted at the end
    M m                              = {1, 1};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 3);
    assert(*r == 2);
  }
  {
    // key already exists and original at the begin
    M m                              = {2, 2, 5, 6};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(*r == 2);
  }
  {
    // key already exists and original in the middle
    M m                              = {0, 2, 2, 4};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r == m.begin() + 3);
    assert(m.size() == 5);
    assert(*r == 2);
  }
  {
    // key already exists and original at the end
    M m                              = {0, 1, 2};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r == m.begin() + 3);
    assert(m.size() == 4);
    assert(*r == 2);
  }
}

template <class KeyContainer>
void test_emplaceable() {
  using M = std::flat_multiset<Emplaceable, std::less<Emplaceable>, KeyContainer>;
  using R = typename M::iterator;

  M m;
  ASSERT_SAME_TYPE(decltype(m.emplace()), R);
  R r = m.emplace(2, 0.0);
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(*r == Emplaceable(2, 0.0));
  r = m.emplace(1, 3.5);
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(*r == Emplaceable(1, 3.5));
  r = m.emplace(1, 3.5);
  assert(r == m.begin() + 1);
  assert(m.size() == 3);
  assert(*r == Emplaceable(1, 3.5));
}

void test() {
  test_one<std::vector<int>>();
  test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();

  test_emplaceable<std::vector<Emplaceable>>();
  test_emplaceable<std::deque<Emplaceable>>();
  test_emplaceable<MinSequenceContainer<Emplaceable>>();
  test_emplaceable<std::vector<Emplaceable, min_allocator<Emplaceable>>>();
}

void test_exception() {
  auto emplace_func = [](auto& m, auto key_arg) { m.emplace(key_arg); };
  test_emplace_exception_guarantee(emplace_func);
}

int main(int, char**) {
  test();
  test_exception();

  return 0;
}
