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
//   iterator emplace_hint(const_iterator position, Args&&... args);

#include <flat_set>
#include <cassert>
#include <deque>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "../../../Emplaceable.h"
#include "DefaultOnly.h"
#include "min_allocator.h"
#include "../helpers.h"

struct CompareTensDigit {
  constexpr bool operator()(auto lhs, auto rhs) const { return (lhs / 10) < (rhs / 10); }
};

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_multiset<Key, std::less<Key>, KeyContainer>;
  using R   = M::iterator;

  {
    // was empty
    M m;
    std::same_as<R> decltype(auto) r = m.emplace_hint(m.end(), typename M::value_type(2));
    assert(r == m.begin());
    assert(m.size() == 1);
    assert(*r == 2);
  }
  {
    // hints correct and no duplicates
    M m                              = {0, 1, 3};
    auto hint                        = m.begin() + 2;
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 4);
    assert(*r == 2);
  }
  {
    // hints correct at the begin
    M m                              = {3, 4};
    auto hint                        = m.begin();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin());
    assert(m.size() == 3);
    assert(*r == 2);
  }
  {
    // hints correct in the middle
    M m                              = {0, 1, 3, 4};
    auto hint                        = m.begin() + 2;
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(*r == 2);
  }
  {
    // hints correct at the end
    M m                              = {0, 1};
    auto hint                        = m.end();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 3);
    assert(*r == 2);
  }
  {
    // hints correct but key already exists
    M m                              = {0, 1, 2, 3, 4};
    auto hint                        = m.begin() + 2;
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 6);
    assert(*r == 2);
  }
  {
    // hint correct and at the first duplicate
    using M2 = std::flat_multiset<Key, CompareTensDigit, KeyContainer>;
    using R2 = M2::iterator;
    M2 m{0, 10, 20, 25, 30};
    auto hint                         = m.begin() + 2;
    std::same_as<R2> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(21));
    assert(r == m.begin() + 2);
    assert(m.size() == 6);
    assert(*r == 21);
  }
  {
    // hint correct and in-between duplicates
    using M2 = std::flat_multiset<Key, CompareTensDigit, KeyContainer>;
    using R2 = M2::iterator;
    M2 m{0, 10, 20, 21, 22, 30};
    auto hint                         = m.begin() + 4;
    std::same_as<R2> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(23));
    assert(r == m.begin() + 4);
    assert(m.size() == 7);
    assert(*r == 23);
    assert(*std::next(r) == 22);
  }
  {
    // hint correct and after duplicates
    using M2 = std::flat_multiset<Key, CompareTensDigit, KeyContainer>;
    using R2 = M2::iterator;
    M2 m{0, 10, 20, 21, 22, 30};
    auto hint                         = m.begin() + 5;
    std::same_as<R2> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(23));
    assert(r == m.begin() + 5);
    assert(m.size() == 7);
    assert(*r == 23);
    assert(*std::next(r) == 30);
  }
  {
    // hints incorrect and no duplicates
    M m                              = {0, 1, 3};
    auto hint                        = m.begin() + 1;
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 4);
    assert(*r == 2);
  }
  {
    // hints incorrectly at the begin
    M m                              = {1, 4};
    auto hint                        = m.begin();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 1);
    assert(m.size() == 3);
    assert(*r == 2);
  }
  {
    // hints incorrectly in the middle
    M m                              = {0, 1, 3, 4};
    auto hint                        = m.begin() + 1;
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(*r == 2);
  }
  {
    // hints incorrectly at the end
    M m                              = {0, 3};
    auto hint                        = m.end();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 1);
    assert(m.size() == 3);
    assert(*r == 2);
  }
  {
    // hints incorrect and key already exists
    M m                              = {0, 1, 2, 3, 4};
    auto hint                        = m.begin();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 6);
    assert(*r == 2);
  }
  {
    // hint incorrect and before the first duplicate
    using M2 = std::flat_multiset<Key, CompareTensDigit, KeyContainer>;
    using R2 = M2::iterator;
    M2 m{0, 10, 20, 21, 22, 30};
    auto hint                         = m.begin();
    std::same_as<R2> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(23));
    assert(r == m.begin() + 2);
    assert(m.size() == 7);
    assert(*r == 23);
  }
  {
    // hint incorrect and after the last duplicate
    using M2 = std::flat_multiset<Key, CompareTensDigit, KeyContainer>;
    using R2 = M2::iterator;
    M2 m{0, 10, 20, 21, 22, 30, 40};
    auto hint                         = m.begin() + 6;
    std::same_as<R2> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(23));
    assert(r == m.begin() + 5);
    assert(m.size() == 8);
    assert(*r == 23);
    assert(*std::next(r) == 30);
  }
}

template <class KeyContainer>
constexpr void test_emplaceable() {
  using M = std::flat_multiset<Emplaceable, std::less<Emplaceable>, KeyContainer>;
  using R = M::iterator;

  M m;
  ASSERT_SAME_TYPE(decltype(m.emplace_hint(m.cbegin())), R);
  R r = m.emplace_hint(m.end(), 2, 0.0);
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(*m.begin() == Emplaceable(2, 0.0));
  r = m.emplace_hint(m.end(), 1, 3.5);
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(*m.begin() == Emplaceable(1, 3.5));
  r = m.emplace_hint(m.end(), 1, 3.5);
  assert(r == m.begin() + 1);
  assert(m.size() == 3);
  assert(*r == Emplaceable(1, 3.5));
}

constexpr bool test() {
  test_one<std::vector<int>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();

  test_emplaceable<std::vector<Emplaceable>>();
  test_emplaceable<std::vector<Emplaceable>>();
  test_emplaceable<MinSequenceContainer<Emplaceable>>();
  test_emplaceable<std::vector<Emplaceable, min_allocator<Emplaceable>>>();

  return true;
}

void test_exception() {
  auto emplace_func = [](auto& m, auto key_arg) { m.emplace_hint(m.begin(), key_arg); };
  test_emplace_exception_guarantee(emplace_func);
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  test_exception();

  return 0;
}
