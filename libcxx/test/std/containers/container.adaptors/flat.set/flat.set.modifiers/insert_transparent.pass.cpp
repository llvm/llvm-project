//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<class K> pair<iterator, bool> insert(K&& x);
// template<class K> iterator insert(const_iterator hint, K&& x);

#include <algorithm>
#include <compare>
#include <concepts>
#include <deque>
#include <flat_set>
#include <functional>
#include <tuple>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type. is_constructible_v<value_type, K> is true.
template <class M, class... Args>
concept CanInsert = requires(M m, Args&&... args) { m.insert(std::forward<Args>(args)...); };

using TransparentSet     = std::flat_set<int, TransparentComparator>;
using TransparentSetIter = typename TransparentSet::iterator;
static_assert(CanInsert<TransparentSet, ExplicitlyConvertibleTransparent<int>>);
static_assert(CanInsert<TransparentSet, TransparentSetIter, ExplicitlyConvertibleTransparent<int>>);
static_assert(!CanInsert<TransparentSet, NonConvertibleTransparent<int>>);
static_assert(!CanInsert<TransparentSet, TransparentSetIter, NonConvertibleTransparent<int>>);

using NonTransparentSet     = std::flat_set<int>;
using NonTransparentSetIter = typename NonTransparentSet::iterator;
static_assert(!CanInsert<NonTransparentSet, ExplicitlyConvertibleTransparent<int>>);
static_assert(!CanInsert<NonTransparentSet, NonTransparentSetIter, ExplicitlyConvertibleTransparent<int>>);
static_assert(!CanInsert<NonTransparentSet, NonConvertibleTransparent<int>>);
static_assert(!CanInsert<NonTransparentSet, NonTransparentSetIter, NonConvertibleTransparent<int>>);

template <class KeyContainer>
void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, TransparentComparator, KeyContainer>;

  {
    const int expected[] = {1, 2, 3, 4, 5};

    {
      // insert(K&&)
      bool transparent_used = false;
      M m{{1, 2, 4, 5}, TransparentComparator{transparent_used}};
      assert(!transparent_used);
      std::same_as<std::pair<typename M::iterator, bool>> decltype(auto) r =
          m.insert(ExplicitlyConvertibleTransparent<Key>{3});
      assert(transparent_used);
      assert(r.first == m.begin() + 2);
      assert(r.second);
      assert(std::ranges::equal(m, expected));
    }
    {
      // insert(const_iterator, K&&)
      bool transparent_used = false;
      M m{{1, 2, 4, 5}, TransparentComparator{transparent_used}};
      assert(!transparent_used);
      std::same_as<typename M::iterator> auto it = m.insert(m.begin(), ExplicitlyConvertibleTransparent<Key>{3});
      assert(transparent_used);
      assert(it == m.begin() + 2);
      assert(std::ranges::equal(m, expected));
    }
  }

  {
    // was empty
    const int expected[] = {3};
    {
      // insert(K&&)
      bool transparent_used = false;
      M m{{}, TransparentComparator{transparent_used}};
      assert(!transparent_used);
      std::same_as<std::pair<typename M::iterator, bool>> decltype(auto) r =
          m.insert(ExplicitlyConvertibleTransparent<Key>{3});
      assert(!transparent_used); // no elements to compare against
      assert(r.first == m.begin());
      assert(r.second);
      assert(std::ranges::equal(m, expected));
    }
    {
      // insert(const_iterator, K&&)
      bool transparent_used = false;
      M m{{}, TransparentComparator{transparent_used}};
      assert(!transparent_used);
      std::same_as<typename M::iterator> auto it = m.insert(m.begin(), ExplicitlyConvertibleTransparent<Key>{3});
      assert(!transparent_used); // no elements to compare against
      assert(it == m.begin());
      assert(std::ranges::equal(m, expected));
    }
  }
}

void test() {
  test_one<std::vector<int>>();
  test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();

  {
    // no ambiguity between insert(pos, P&&) and insert(first, last)
    using M = std::flat_set<int>;
    struct Evil {
      operator M::value_type() const;
      operator M::const_iterator() const;
    };
    std::flat_set<int> m;
    ASSERT_SAME_TYPE(decltype(m.insert(Evil())), std::pair<M::iterator, bool>);
    ASSERT_SAME_TYPE(decltype(m.insert(m.begin(), Evil())), M::iterator);
    ASSERT_SAME_TYPE(decltype(m.insert(m.begin(), m.end())), void);
  }
}

void test_exception() {
  {
    auto insert_func = [](auto& m, auto key_arg) {
      using FlatSet = std::decay_t<decltype(m)>;
      m.insert(ExplicitlyConvertibleTransparent<typename FlatSet::key_type>{key_arg});
    };
    test_emplace_exception_guarantee(insert_func);
  }
  {
    auto insert_func_iter = [](auto& m, auto key_arg) {
      using FlatSet = std::decay_t<decltype(m)>;
      m.insert(m.begin(), ExplicitlyConvertibleTransparent<typename FlatSet::key_type>{key_arg});
    };
    test_emplace_exception_guarantee(insert_func_iter);
  }
}

int main(int, char**) {
  test();
  test_exception();

  return 0;
}
