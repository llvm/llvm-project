//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set(flat_set&&);

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
    using M = std::flat_set<int, C, KeyContainer<int, A>>;
    M mo    = M({1, 2, 3}, C(5), A(7));
    M m     = std::move(mo);
    assert((m == M{1, 2, 3}));
    assert(m.key_comp() == C(5));
    assert(std::move(m).extract().get_allocator() == A(7));

    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(std::move(mo).extract().get_allocator().get_id() == test_alloc_base::moved_value);
  }
  {
    using C = test_less<int>;
    using A = min_allocator<int>;
    using M = std::flat_set<int, C, KeyContainer<int, A>>;
    M mo    = M({1, 2, 3}, C(5), A());
    M m     = std::move(mo);
    assert((m == M{1, 2, 3}));
    assert(m.key_comp() == C(5));
    assert(std::move(m).extract().get_allocator() == A());

    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(std::move(mo).extract().get_allocator() == A());
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    // A moved-from flat_set maintains its class invariant in the presence of moved-from comparators.
    using M = std::flat_set<int, std::function<bool(int, int)>, KeyContainer<int>>;
    M mo    = M({1, 2, 3}, std::less<int>());
    M m     = std::move(mo);
    assert(m.size() == 3);
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));
    assert(m.key_comp()(1, 2) == true);

    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    LIBCPP_ASSERT(m.key_comp()(1, 2) == true);
    LIBCPP_ASSERT(mo.empty());
    mo.insert({1, 2, 3}); // insert has no preconditions
    assert(m == mo);
  }
  {
    // moved-from object maintains invariant if the underlying container does not clear after move
    using M = std::flat_set<int, std::less<>, CopyOnlyVector<int>>;
    M m1    = M({1, 2, 3});
    M m2    = std::move(m1);
    assert(m2.size() == 3);
    check_invariant(m1);
    LIBCPP_ASSERT(m1.empty());
    LIBCPP_ASSERT(m1.size() == 0);
  }
}

template <class T>
struct ThrowingMoveAllocator {
  using value_type                                    = T;
  explicit ThrowingMoveAllocator()                    = default;
  ThrowingMoveAllocator(const ThrowingMoveAllocator&) = default;
  constexpr ThrowingMoveAllocator(ThrowingMoveAllocator&&) noexcept(false) {}
  constexpr T* allocate(std::ptrdiff_t n) { return std::allocator<T>().allocate(n); }
  constexpr void deallocate(T* p, std::ptrdiff_t n) { return std::allocator<T>().deallocate(p, n); }
  friend bool operator==(ThrowingMoveAllocator, ThrowingMoveAllocator) = default;
};

struct ThrowingMoveComp {
  ThrowingMoveComp() = default;
  constexpr ThrowingMoveComp(const ThrowingMoveComp&) noexcept(true) {}
  constexpr ThrowingMoveComp(ThrowingMoveComp&&) noexcept(false) {}
  constexpr bool operator()(const auto&, const auto&) const { return false; }
};

template <template <class...> class KeyContainer>
constexpr void test_move_noexcept() {
  {
    using C = std::flat_set<int, std::less<int>, KeyContainer<int>>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
  {
    using C = std::flat_set<int, std::less<int>, KeyContainer<int, test_allocator<int>>>;
    LIBCPP_STATIC_ASSERT(std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
#if _LIBCPP_VERSION
  if (!TEST_IS_CONSTANT_EVALUATED) {
    // Container fails to be nothrow-move-constructible; this relies on libc++'s support for non-nothrow-copyable allocators
    using C = std::flat_set<int, std::less<int>, std::deque<int, ThrowingMoveAllocator<int>>>;
    static_assert(!std::is_nothrow_move_constructible_v<std::deque<int, ThrowingMoveAllocator<int>>>);
    static_assert(!std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
#endif // _LIBCPP_VERSION
  {
    // Comparator fails to be nothrow-move-constructible
    using C = std::flat_set<int, ThrowingMoveComp, KeyContainer<int>>;
    static_assert(!std::is_nothrow_move_constructible_v<C>);
    C c;
    C d = std::move(c);
  }
}

constexpr bool test() {
  test<std::vector>();
  test_move_noexcept<std::vector>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque>();
    test_move_noexcept<std::deque>();
  }
  return true;
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
    using M   = std::flat_set<int, std::less<int>, EvilContainer>;
    M mo      = {1, 2, 3};
    countdown = 1;
    try {
      M m = std::move(mo);
      assert(false); // not reached
    } catch (int x) {
      assert(x == 42);
    }
    // The source flat_set maintains its class invariant.
    check_invariant(mo);
    LIBCPP_ASSERT(mo.empty());
  }
}
#endif // !defined(TEST_HAS_NO_EXCEPTIONS)

int main(int, char**) {
  test();
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  test_move_exception();
#endif
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
