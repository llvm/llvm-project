//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template <class... Args>
//   pair<iterator, bool> emplace(Args&&... args);

#include <flat_map>
#include <cassert>
#include <deque>
#include <tuple>

#include "../helpers.h"
#include "test_macros.h"
#include "../../../Emplaceable.h"
#include "DefaultOnly.h"
#include "min_allocator.h"

// Constraints: is_constructible_v<pair<key_type, mapped_type>, Args...> is true.
template <class M, class... Args>
concept CanEmplace = requires(M m, Args&&... args) { m.emplace(std::forward<Args>(args)...); };

using Map = std::flat_map<Emplaceable, Emplaceable>;
static_assert(CanEmplace<Map>);
static_assert(CanEmplace<Map, Emplaceable, Emplaceable>);
static_assert(CanEmplace<Map, std::piecewise_construct_t, std::tuple<int, double>, std::tuple<int, double>>);
static_assert(!CanEmplace<Map, Emplaceable>);
static_assert(!CanEmplace<Map, int, double>);

int main(int, char**) {
  {
    // Emplaceable
    using M = std::flat_map<int, Emplaceable>;
    using R = std::pair<M::iterator, bool>;
    M m;
    ASSERT_SAME_TYPE(decltype(m.emplace()), R);
    R r = m.emplace(std::piecewise_construct, std::forward_as_tuple(2), std::forward_as_tuple());
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(m.begin()->first == 2);
    assert(m.begin()->second == Emplaceable());
    r = m.emplace(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 2);
    assert(m.begin()->first == 1);
    assert(m.begin()->second == Emplaceable(2, 3.5));
    r = m.emplace(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
    assert(!r.second);
    assert(r.first == m.begin());
    assert(m.size() == 2);
    assert(m.begin()->first == 1);
    assert(m.begin()->second == Emplaceable(2, 3.5));
  }
  {
    using M = std::flat_map<int, double>;
    using R = std::pair<M::iterator, bool>;
    M m;
    ASSERT_SAME_TYPE(decltype(m.emplace()), R);
    R r = m.emplace(M::value_type(2, 3.5));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(m.begin()->first == 2);
    assert(m.begin()->second == 3.5);
  }
  {
    using M =
        std::flat_map<int,
                      Emplaceable,
                      std::less<int>,
                      std::vector<int, min_allocator<int>>,
                      std::vector<Emplaceable, min_allocator<Emplaceable>>>;
    using R = std::pair<M::iterator, bool>;
    M m;
    ASSERT_SAME_TYPE(decltype(m.emplace()), R);
    R r = m.emplace(std::piecewise_construct, std::forward_as_tuple(2), std::forward_as_tuple());
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(m.begin()->first == 2);
    assert(m.begin()->second == Emplaceable());
    r = m.emplace(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 2);
    assert(m.begin()->first == 1);
    assert(m.begin()->second == Emplaceable(2, 3.5));
    r = m.emplace(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
    assert(!r.second);
    assert(r.first == m.begin());
    assert(m.size() == 2);
    assert(m.begin()->first == 1);
    assert(m.begin()->second == Emplaceable(2, 3.5));
  }
  {
    using M =
        std::flat_map< int,
                       double,
                       std::less<int>,
                       std::deque<int, min_allocator<int>>,
                       std::deque<double, min_allocator<double>>>;
    using R = std::pair<M::iterator, bool>;
    M m;
    R r = m.emplace(M::value_type(2, 3.5));
    ASSERT_SAME_TYPE(decltype(m.emplace()), R);
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(m.begin()->first == 2);
    assert(m.begin()->second == 3.5);
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    // Throw on emplace the key, and underlying has strong exception guarantee
    using M = std::flat_map<CopyableThrowOnMove, int>;
    std::vector<CopyableThrowOnMove> a;
    a.reserve(4);
    a.emplace_back(1);
    a.emplace_back(2);
    a.emplace_back(3);
    a.emplace_back(4);

    std::vector<int> b                    = {5, 6, 7, 8};
    [[maybe_unused]] auto expected_keys   = a;
    [[maybe_unused]] auto expected_values = b;
    M m(std::sorted_unique, std::move(a), std::move(b));

    try {
      m.emplace(std::piecewise_construct, std::tuple(0), std::tuple(0));
      assert(false);
    } catch (int) {
      assert(m.keys().size() == m.values().size());
      assert(is_sorted_and_unique(m.keys()));
      // In libc++, the flat_map is unchanged
      LIBCPP_ASSERT(m.size() == 4);
      LIBCPP_ASSERT(m.keys() == expected_keys);
      LIBCPP_ASSERT(m.values() == expected_values);
    }
  }
  {
    // Throw on emplace the key, and underlying has no strong exception guarantee
    using M = std::flat_map<MoveOnlyThrowOnMove, int>;
    std::vector<MoveOnlyThrowOnMove> a;
    a.reserve(4);
    a.emplace_back(1);
    a.emplace_back(2);
    a.emplace_back(3);
    a.emplace_back(4);

    std::vector<int> b = {5, 6, 7, 8};
    M m(std::sorted_unique, std::move(a), std::move(b));
    try {
      m.emplace(std::piecewise_construct, std::tuple(0), std::tuple(0));
      assert(false);
    } catch (int) {
      assert(m.keys().size() == m.values().size());
      assert(is_sorted_and_unique(m.keys()));
      // In libc++, the flat_map is cleared
      LIBCPP_ASSERT(m.size() == 0);
    }
  }
  {
    // Throw on emplace the value, and underlying has strong exception guarantee
    using M            = std::flat_map<int, CopyableThrowOnMove>;
    std::vector<int> a = {1, 2, 3, 4};
    std::vector<CopyableThrowOnMove> b;
    b.reserve(4);
    b.emplace_back(1);
    b.emplace_back(2);
    b.emplace_back(3);
    b.emplace_back(4);

    [[maybe_unused]] auto expected_keys   = a;
    [[maybe_unused]] auto expected_values = b;
    M m(std::sorted_unique, std::move(a), std::move(b));

    try {
      m.emplace(std::piecewise_construct, std::tuple(0), std::tuple(0));
      assert(false);
    } catch (int) {
      assert(m.keys().size() == m.values().size());
      assert(is_sorted_and_unique(m.keys()));
      // In libc++, the emplaced key is erased and the flat_map is unchanged
      LIBCPP_ASSERT(m.size() == 4);
      LIBCPP_ASSERT(m.keys() == expected_keys);
      LIBCPP_ASSERT(m.values() == expected_values);
    }
  }
  {
    // Throw on emplace the value, and underlying has no strong exception guarantee
    using M            = std::flat_map<int, MoveOnlyThrowOnMove>;
    std::vector<int> a = {1, 2, 3, 4};
    std::vector<MoveOnlyThrowOnMove> b;
    b.reserve(4);
    b.emplace_back(1);
    b.emplace_back(2);
    b.emplace_back(3);
    b.emplace_back(4);

    M m(std::sorted_unique, std::move(a), std::move(b));

    try {
      m.emplace(std::piecewise_construct, std::tuple(0), std::tuple(0));
      assert(false);
    } catch (int) {
      assert(m.keys().size() == m.values().size());
      assert(is_sorted_and_unique(m.keys()));
      // In libc++, the flat_map is unchanged
      LIBCPP_ASSERT(m.size() == 0);
    }
  }
  {
    using M = std::flat_map<ThrowOnSecondMove, CopyableThrowOnMove>;
    std::vector<ThrowOnSecondMove> a;
    a.reserve(4);
    a.emplace_back(1);
    a.emplace_back(2);
    a.emplace_back(3);
    a.emplace_back(4);

    std::vector<CopyableThrowOnMove> b;
    b.reserve(4);
    b.emplace_back(1);
    b.emplace_back(2);
    b.emplace_back(3);
    b.emplace_back(4);

    M m(std::sorted_unique, std::move(a), std::move(b));
    try {
      m.emplace(std::piecewise_construct, std::tuple(0), std::tuple(0));
      assert(false);
    } catch (int) {
      assert(m.keys().size() == m.values().size());
      assert(is_sorted_and_unique(m.keys()));
      // In libc++, we try to erase the key after value emplacement failure.
      // and after erasure failure, we clear the flat_map
      LIBCPP_ASSERT(m.size() == 0);
    }
  }
#endif

  return 0;
}
