//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// class flat_multimap

// template <class... Args>
// iterator emplace(Args&&... args);

#include <flat_map>
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

// Constraints: is_constructible_v<pair<key_type, mapped_type>, Args...> is true.
template <class M, class... Args>
concept CanEmplace = requires(M m, Args&&... args) { m.emplace(std::forward<Args>(args)...); };

using Map = std::flat_multimap<Emplaceable, Emplaceable>;
static_assert(CanEmplace<Map>);
static_assert(CanEmplace<Map, Emplaceable, Emplaceable>);
static_assert(CanEmplace<Map, std::piecewise_construct_t, std::tuple<int, double>, std::tuple<int, double>>);
static_assert(!CanEmplace<Map, Emplaceable>);
static_assert(!CanEmplace<Map, int, double>);

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using R     = typename M::iterator;

  {
    // was empty
    M m;
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2, 3.5));
    assert(r == m.begin());
    assert(m.size() == 1);
    assert(r->first == 2);
    assert(r->second == 3.5);
  }
  {
    // key does not exist and inserted at the begin
    M m                              = {{3, 4.0}, {3, 3.0}, {3, 1.0}, {7, 0.0}};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2, 2.0));
    assert(r == m.begin());
    assert(m.size() == 5);
    assert(r->first == 2);
    assert(r->second == 2.0);
  }
  {
    // key does not exist and inserted in the middle
    M m                              = {{1, 4.0}, {1, 3.0}, {3, 1.0}, {4, 0.0}};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2, 2.0));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(r->first == 2);
    assert(r->second == 2.0);
  }
  {
    // key does not exist and inserted at the end
    M m                              = {{1, 4.0}, {1, 3.0}};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2, 2.0));
    assert(r == m.begin() + 2);
    assert(m.size() == 3);
    assert(r->first == 2);
    assert(r->second == 2.0);
  }
  {
    // key already exists and original at the begin
    M m                              = {{2, 4.0}, {2, 3.0}, {5, 1.0}, {6, 0.0}};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2, 2.0));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(r->first == 2);
    assert(r->second == 2.0);
  }
  {
    // key already exists and original in the middle
    M m                              = {{0, 4.0}, {2, 3.0}, {2, 1.0}, {4, 0.0}};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2, 2.0));
    assert(r == m.begin() + 3);
    assert(m.size() == 5);
    assert(r->first == 2);
    assert(r->second == 2.0);
  }
  {
    // key already exists and original at the end
    M m                              = {{0, 4.0}, {1, 3.0}, {2, 1.0}};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2, 2.0));
    assert(r == m.begin() + 3);
    assert(m.size() == 4);
    assert(r->first == 2);
    assert(r->second == 2.0);
  }
}

template <class KeyContainer, class ValueContainer>
constexpr void test_emplaceable() {
  using M = std::flat_multimap<int, Emplaceable, std::less<int>, KeyContainer, ValueContainer>;
  using R = typename M::iterator;

  M m;
  std::same_as<R> decltype(auto) r =
      m.emplace(std::piecewise_construct, std::forward_as_tuple(2), std::forward_as_tuple());
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(m.begin()->first == 2);
  assert(m.begin()->second == Emplaceable());
  r = m.emplace(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(m.begin()->first == 1);
  assert(m.begin()->second == Emplaceable(2, 3.5));
  r = m.emplace(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
  assert(r == m.begin() + 1);
  assert(m.size() == 3);
  assert(m.begin()->first == 1);
  assert(m.begin()->second == Emplaceable(2, 3.5));
}

constexpr bool test() {
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque<int>, std::vector<double>>();
    test_emplaceable<std::deque<int>, std::vector<Emplaceable>>();
  }

  test<std::vector<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  test_emplaceable<std::vector<int>, std::vector<Emplaceable>>();
  test_emplaceable<MinSequenceContainer<int>, MinSequenceContainer<Emplaceable>>();
  test_emplaceable<std::vector<int, min_allocator<int>>, std::vector<Emplaceable, min_allocator<Emplaceable>>>();

  if (!TEST_IS_CONSTANT_EVALUATED) {
    auto emplace_func = [](auto& m, auto key_arg, auto value_arg) {
      m.emplace(std::piecewise_construct, std::tuple(key_arg), std::tuple(value_arg));
    };
    test_emplace_exception_guarantee(emplace_func);
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
