//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_multiset(flat_multiset&&);

#include <algorithm>
#include <deque>
#include <flat_set>
#include <functional>
#include <utility>
#include <vector>

#include "../helpers.h"
#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <template <class...> class KeyContainer>
constexpr void test() {
  {
    using C = test_less<int>;
    using A = test_allocator<int>;
    using M = std::flat_multiset<int, C, KeyContainer<int, A>>;
    M mo    = M({1, 2, 1, 3}, C(5), A(7));
    M m     = std::move(mo);
    assert((m == M{1, 1, 2, 3}));
    assert(m.key_comp() == C(5));
    assert(std::move(m).extract().get_allocator() == A(7));

    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(std::move(mo).extract().get_allocator().get_id() == test_alloc_base::moved_value);
  }
  {
    using C = test_less<int>;
    using A = min_allocator<int>;
    using M = std::flat_multiset<int, C, KeyContainer<int, A>>;
    M mo    = M({1, 2, 1, 3}, C(5), A());
    M m     = std::move(mo);
    assert((m == M{1, 1, 2, 3}));
    assert(m.key_comp() == C(5));
    assert(std::move(m).extract().get_allocator() == A());

    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(std::move(mo).extract().get_allocator() == A());
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    // A moved-from flat_multiset maintains its class invariant in the presence of moved-from comparators.
    using M = std::flat_multiset<int, std::function<bool(int, int)>, KeyContainer<int>>;
    M mo    = M({1, 2, 1, 3}, std::less<int>());
    M m     = std::move(mo);
    assert(m.size() == 4);
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));
    assert(m.key_comp()(1, 2) == true);

    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    LIBCPP_ASSERT(m.key_comp()(1, 2) == true);
    LIBCPP_ASSERT(mo.empty());
    mo.insert({1, 1, 2, 3}); // insert has no preconditions
    assert(m == mo);
  }
  {
    // moved-from object maintains invariant if the underlying container does not clear after move
    using M = std::flat_multiset<int, std::less<>, CopyOnlyVector<int>>;
    M m1    = M({1, 2, 1, 3});
    M m2    = std::move(m1);
    assert(m2.size() == 4);
    check_invariant(m1);
    LIBCPP_ASSERT(m1.empty());
    LIBCPP_ASSERT(m1.size() == 0);
  }
}

constexpr bool test() {
  test<std::vector>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque>();

  return true;
}

template <class T>
struct ThrowingMoveAllocator {
  using value_type                                    = T;
  explicit ThrowingMoveAllocator()                    = default;
  ThrowingMoveAllocator(const ThrowingMoveAllocator&) = default;
  ThrowingMoveAllocator(ThrowingMoveAllocator&&) noexcept(false) {}
  T* allocate(std::ptrdiff_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* p, std::ptrdiff_t n) { return std::allocator<T>().deallocate(p, n); }
  friend bool operator==(ThrowingMoveAllocator, ThrowingMoveAllocator) = default;
};

struct ThrowingMoveComp {
  ThrowingMoveComp() = default;
  ThrowingMoveComp(const ThrowingMoveComp&) noexcept(true) {}
  ThrowingMoveComp(ThrowingMoveComp&&) noexcept(false) {}
  bool operator()(const auto&, const auto&) const { return false; }
};

struct MoveSensitiveComp {
  MoveSensitiveComp() noexcept(false)                  = default;
  MoveSensitiveComp(const MoveSensitiveComp&) noexcept = default;
  MoveSensitiveComp(MoveSensitiveComp&& rhs) { rhs.is_moved_from_ = true; }
  MoveSensitiveComp& operator=(const MoveSensitiveComp&) noexcept(false) = default;
  MoveSensitiveComp& operator=(MoveSensitiveComp&& rhs) {
    rhs.is_moved_from_ = true;
    return *this;
  }
  bool operator()(const auto&, const auto&) const { return false; }
  bool is_moved_from_ = false;
};

void test_move_noexcept() {
  {
    using C = std::flat_multiset<int>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
  {
    using C = std::flat_multiset<int, std::less<int>, std::deque<int, test_allocator<int>>>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
#if _LIBCPP_VERSION
  {
    // Container fails to be nothrow-move-constructible; this relies on libc++'s support for non-nothrow-copyable allocators
    using C = std::flat_multiset<int, std::less<int>, std::deque<int, ThrowingMoveAllocator<int>>>;
    static_assert(!std::is_nothrow_move_constructible_v<std::deque<int, ThrowingMoveAllocator<int>>>);
    static_assert(!std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
#endif // _LIBCPP_VERSION
  {
    // Comparator fails to be nothrow-move-constructible
    using C = std::flat_multiset<int, ThrowingMoveComp>;
    static_assert(!std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
}

#if !defined(TEST_HAS_NO_EXCEPTIONS)
static int countdown = 0;

struct EvilContainer : std::vector<int> {
  EvilContainer() = default;
  EvilContainer(EvilContainer&& rhs) {
    // Throw on move-construction.
    if (--countdown == 0) {
      rhs.insert(rhs.end(), 0);
      rhs.insert(rhs.end(), 0);
      throw 42;
    }
  }
};

void test_move_exception() {
  {
    using M   = std::flat_multiset<int, std::less<int>, EvilContainer>;
    M mo      = {1, 2, 3};
    countdown = 1;
    try {
      M m = std::move(mo);
      assert(false); // not reached
    } catch (int x) {
      assert(x == 42);
    }
    // The source flat_multiset maintains its class invariant.
    check_invariant(mo);
    LIBCPP_ASSERT(mo.empty());
  }
}
#endif // !defined(TEST_HAS_NO_EXCEPTIONS)

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  test_move_noexcept();
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  test_move_exception();
#endif // !defined(TEST_HAS_NO_EXCEPTIONS)

  return 0;
}
