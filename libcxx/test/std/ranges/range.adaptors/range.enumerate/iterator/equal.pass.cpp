//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// class enumerate_view

// class enumerate_view::iterator

// friend constexpr bool operator==(const iterator& x, const iterator& y) noexcept;

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "../types.h"
// #include "../types_iterators.h"

// template <bool Const>
// struct Iterator {
//   using value_type       = int
//   using difference_type  = std::std::ptrdiff_t;
//   using iterator_concept = std::input_iterator_tag;

//   constexpr decltype(auto) operator*() const { return *it_; }
//   constexpr Iterator& operator++() {
//     ++it_;

//     return *this;
//   }
//   constexpr void operator++(int) { ++it_; }

//   std::tuple<std::ptrdiff_t, int>* it_;
// };

// template <bool Const>
// struct Sentinel {
//   constexpr bool operator==(const Iterator<Const>& i) const { return i.it_ == end_; }

//   std::tuple<std::ptrdiff_t, int>* end_;
// };

// template <bool Const>
// struct CrossComparableSentinel {
//   template <bool C>
//   constexpr bool operator==(const Iterator<C>& i) const {
//     return i.it_ == end_;
//   }

//   std::tuple<std::ptrdiff_t, int>* end_;
// };

constexpr bool test() {
  int buff[] = {0, 1, 2, 3, 5};
  {
    using View = std::ranges::enumerate_view<RangeView>;
    RangeView const range(buff, buff + 5);

    std::same_as<View> decltype(auto) ev = std::views::enumerate(range);

    auto it1 = ev.begin();
    auto it2 = it1 + 5;

    assert(it1 == it1);
    ASSERT_NOEXCEPT(it1 == it1);
    assert(it1 != it2);
    ASSERT_NOEXCEPT(it1 != it2);
    assert(it2 != it1);
    ASSERT_NOEXCEPT(it2 != it1);
    assert(it2 == ev.end());
    assert(ev.end() == it2);

    for (std::size_t index = 0; index != 5; ++index) {
      ++it1;
    }

    assert(it1 == it2);
    assert(it2 == it1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
