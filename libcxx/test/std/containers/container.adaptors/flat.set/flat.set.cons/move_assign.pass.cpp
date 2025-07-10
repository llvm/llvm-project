//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set& operator=(flat_set&&);

#include <algorithm>
#include <deque>
#include <flat_set>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "test_macros.h"
#include "MoveOnly.h"
#include "../helpers.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

struct MoveNegates {
  int value_    = 0;
  MoveNegates() = default;
  constexpr MoveNegates(int v) : value_(v) {}
  constexpr MoveNegates(MoveNegates&& rhs) : value_(rhs.value_) { rhs.value_ = -rhs.value_; }
  constexpr MoveNegates& operator=(MoveNegates&& rhs) {
    value_     = rhs.value_;
    rhs.value_ = -rhs.value_;
    return *this;
  }
  ~MoveNegates()                             = default;
  auto operator<=>(const MoveNegates&) const = default;
};

struct MoveClears {
  int value_   = 0;
  MoveClears() = default;
  constexpr MoveClears(int v) : value_(v) {}
  constexpr MoveClears(MoveClears&& rhs) : value_(rhs.value_) { rhs.value_ = 0; }
  constexpr MoveClears& operator=(MoveClears&& rhs) {
    value_     = rhs.value_;
    rhs.value_ = 0;
    return *this;
  }
  ~MoveClears()                             = default;
  auto operator<=>(const MoveClears&) const = default;
};

#if !defined(TEST_HAS_NO_EXCEPTIONS)
struct MoveAssignThrows : std::vector<int> {
  using std::vector<int>::vector;
  MoveAssignThrows& operator=(MoveAssignThrows&& other) {
    push_back(0);
    push_back(0);
    other.push_back(0);
    other.push_back(0);
    throw 42;
  }
};
#endif // TEST_HAS_NO_EXCEPTIONS

template <template <class...> class KeyContainer>
constexpr void test_move_assign_clears() {
  // Preserves the class invariant for the moved-from flat_set.
  {
    const int expected[] = {1, 2, 3, 4, 5, 6, 7, 8};
    using M              = std::flat_set<MoveNegates, std::less<MoveNegates>, KeyContainer<MoveNegates>>;
    M m                  = M(expected, expected + 8);
    M m2                 = M(expected, expected + 3);

    m2 = std::move(m);

    assert(std::equal(m2.begin(), m2.end(), expected, expected + 8));
    LIBCPP_ASSERT(m.empty());
    assert(std::is_sorted(m.begin(), m.end(), m.key_comp()));                // still sorted
    assert(std::adjacent_find(m.begin(), m.end(), m.key_comp()) == m.end()); // still contains no duplicates
    m.insert(1);
    m.insert(2);
    assert(m.contains(1));
    assert(m.find(2) != m.end());
  }
  {
    const int expected[] = {1, 2, 3, 4, 5, 6, 7, 8};
    using M              = std::flat_set<MoveClears, std::less<MoveClears>, KeyContainer<MoveClears>>;
    M m                  = M(expected, expected + 8);
    M m2                 = M(expected, expected + 3);

    m2 = std::move(m);

    assert(std::equal(m2.begin(), m2.end(), expected, expected + 8));
    LIBCPP_ASSERT(m.empty());
    assert(std::is_sorted(m.begin(), m.end(), m.key_comp()));                // still sorted
    assert(std::adjacent_find(m.begin(), m.end(), m.key_comp()) == m.end()); // still contains no duplicates
    m.insert(1);
    m.insert(2);
    assert(m.contains(1));
    assert(m.find(2) != m.end());
  }
  {
    // moved-from object maintains invariant if one of underlying container does not clear after move
    using M = std::flat_set<int, std::less<>, CopyOnlyVector<int>>;
    M m1    = M({1, 2, 3});
    M m2    = M({1, 2});
    m2      = std::move(m1);
    assert(m2.size() == 3);
    check_invariant(m1);
    LIBCPP_ASSERT(m1.empty());
  }
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  if (!TEST_IS_CONSTANT_EVALUATED) {
    using M = std::flat_set<int, std::less<>, MoveAssignThrows>;
    M m1    = {1, 2, 3};
    M m2    = {1, 2};
    try {
      m2 = std::move(m1);
      assert(false);
    } catch (int e) {
      assert(e == 42);
    }
    check_invariant(m1);
    check_invariant(m2);
    LIBCPP_ASSERT(m1.empty());
    LIBCPP_ASSERT(m2.empty());
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

struct MoveSensitiveComp {
  MoveSensitiveComp() noexcept(false)                         = default;
  MoveSensitiveComp(const MoveSensitiveComp&) noexcept(false) = default;
  MoveSensitiveComp(MoveSensitiveComp&& rhs) { rhs.is_moved_from_ = true; }
  MoveSensitiveComp& operator=(const MoveSensitiveComp&) noexcept = default;
  MoveSensitiveComp& operator=(MoveSensitiveComp&& rhs) {
    rhs.is_moved_from_ = true;
    return *this;
  }
  bool operator()(const auto&, const auto&) const { return false; }
  bool is_moved_from_ = false;
};

struct MoveThrowsComp {
  MoveThrowsComp(MoveThrowsComp&&) noexcept(false);
  MoveThrowsComp(const MoveThrowsComp&) noexcept(true);
  MoveThrowsComp& operator=(MoveThrowsComp&&) noexcept(false);
  MoveThrowsComp& operator=(const MoveThrowsComp&) noexcept(true);
  bool operator()(const auto&, const auto&) const;
};

void test_move_assign_no_except() {
  // This tests a conforming extension

  {
    using C = std::flat_set<int, int>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_assignable_v<C>);
  }
  {
    using C = std::flat_set<MoveOnly, std::less<MoveOnly>, std::vector<MoveOnly, test_allocator<MoveOnly>>>;
    static_assert(!std::is_nothrow_move_assignable_v<C>);
  }
  {
    using C = std::flat_set<int, std::less<int>, std::vector<int, test_allocator<int>>>;
    static_assert(!std::is_nothrow_move_assignable_v<C>);
  }
  {
    using C = std::flat_set<MoveOnly, std::less<MoveOnly>, std::vector<MoveOnly, other_allocator<MoveOnly>>>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_assignable_v<C>);
  }
  {
    using C = std::flat_set<int, std::less<int>, std::vector<int, other_allocator<int>>>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_assignable_v<C>);
  }
  {
    // Test with a comparator that throws on move-assignment.
    using C = std::flat_set<int, MoveThrowsComp>;
    LIBCPP_STATIC_ASSERT(!std::is_nothrow_move_assignable_v<C>);
  }
  {
    // Test with a container that throws on move-assignment.
    using C = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
    static_assert(!std::is_nothrow_move_assignable_v<C>);
  }
}

template <template <class...> class KeyContainer>
constexpr void test() {
  {
    using C                           = test_less<int>;
    using A1                          = test_allocator<int>;
    using M                           = std::flat_set<int, C, KeyContainer<int, A1>>;
    M mo                              = M({1, 2, 3}, C(5), A1(7));
    M m                               = M({}, C(3), A1(7));
    std::same_as<M&> decltype(auto) r = m = std::move(mo);
    assert(&r == &m);
    assert((m == M{1, 2, 3}));
    assert(m.key_comp() == C(5));
    auto ks = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(mo.empty());
  }
  {
    using C                           = test_less<int>;
    using A1                          = other_allocator<int>;
    using M                           = std::flat_set<int, C, KeyContainer<int, A1>>;
    M mo                              = M({4, 5}, C(5), A1(7));
    M m                               = M({1, 2, 3, 4}, C(3), A1(7));
    std::same_as<M&> decltype(auto) r = m = std::move(mo);
    assert(&r == &m);
    assert((m == M{4, 5}));
    assert(m.key_comp() == C(5));
    auto ks = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(mo.empty());
  }
  {
    using A                           = min_allocator<int>;
    using M                           = std::flat_set<int, std::greater<int>, KeyContainer<int, A>>;
    M mo                              = M({5, 4, 3}, A());
    M m                               = M({4, 3, 2, 1}, A());
    std::same_as<M&> decltype(auto) r = m = std::move(mo);
    assert(&r == &m);
    assert((m == M{5, 4, 3}));
    auto ks = std::move(m).extract();
    assert(ks.get_allocator() == A());
    assert(mo.empty());
  }
}

constexpr bool test() {
  test<std::vector>();
  test_move_assign_clears<std::vector>();

#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque>();
    test_move_assign_clears<std::deque>();
  }

  return true;
}

int main(int, char**) {
  test();
  test_move_assign_no_except();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
