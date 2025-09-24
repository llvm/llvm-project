//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//   template<input_range V>
//     requires view<V>
//   template<bool Const>
//   class to_input_view<V>::iterator

//    iterator() requires default_initializable<iterator_t<Base>> = default;

#include <cassert>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"

struct DefaultInitializableIterator {
  int i_; // Deliberately uninitialized.

  using difference_type   = std::intptr_t;
  using value_type        = int;
  using iterator_category = std::random_access_iterator_tag;

  constexpr int operator*() const { return i_; }

  constexpr DefaultInitializableIterator& operator++() { return *this; }
  constexpr void operator++(int) {}

  friend constexpr bool operator==(const DefaultInitializableIterator&, const DefaultInitializableIterator&) = default;
};

static_assert(std::default_initializable<DefaultInitializableIterator>);

struct DefaultInitializableIteratorView : std::ranges::view_base {
  DefaultInitializableIterator begin() const;
  DefaultInitializableIterator end() const;
};

struct NonDefaultInitializableIteratorView : std::ranges::view_base {
  static_assert(!std::default_initializable<cpp20_input_iterator<int*>>);

  cpp20_input_iterator<int*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<int*>> end() const;
};

template <class View>
using CacheLatestViewSentinelT = std::ranges::sentinel_t<std::ranges::cache_latest_view<View>>;

// Check that the to_input_view's iterator is default initializable when the underlying iterator is.
static_assert(std::default_initializable<CacheLatestViewSentinelT<DefaultInitializableIteratorView>>);
static_assert(!std::default_initializable<CacheLatestViewSentinelT<NonDefaultInitializableIteratorView>>);

constexpr bool test() {
  {
    CacheLatestViewSentinelT<DefaultInitializableIteratorView> it;
    assert(*it == 0); // DefaultInitializableIterator has to be initialized to have value 0.
  }
  {
    CacheLatestViewSentinelT<DefaultInitializableIteratorView> it = {};
    assert(*it == 0); // DefaultInitializableIterator has to be initialized to have value 0.
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
