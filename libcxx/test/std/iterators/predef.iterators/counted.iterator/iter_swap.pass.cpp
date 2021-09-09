//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<indirectly_swappable<I> I2>
//   friend constexpr void
//     iter_swap(const counted_iterator& x, const counted_iterator<I2>& y)
//       noexcept(noexcept(ranges::iter_swap(x.current, y.current)));

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

template<bool IsNoexcept>
class HasNoexceptIterSwap
{
  int *it_;

public:
  typedef          std::input_iterator_tag                      iterator_category;
  typedef int                                                   value_type;
  typedef typename std::iterator_traits<int *>::difference_type difference_type;
  typedef int *                                                 pointer;
  typedef int &                                                 reference;

  constexpr int *base() const {return it_;}

  HasNoexceptIterSwap() = default;
  explicit constexpr HasNoexceptIterSwap(int *it) : it_(it) {}

  constexpr reference operator*() const {return *it_;}

  constexpr HasNoexceptIterSwap& operator++() {++it_; return *this;}
  constexpr HasNoexceptIterSwap operator++(int)
      {HasNoexceptIterSwap tmp(*this); ++(*this); return tmp;}

  friend void iter_swap(
    const HasNoexceptIterSwap&, const HasNoexceptIterSwap&) noexcept(IsNoexcept) {}
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1 = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = std::counted_iterator<decltype(iter1)>(iter1, 8);
    auto commonIter2 = std::counted_iterator<decltype(iter1)>(iter1, 8);
    for (auto i = 0; i < 4; ++i) ++commonIter2;
    assert(*commonIter2 == 5);
    std::ranges::iter_swap(commonIter1, commonIter2);
    assert(*commonIter1 == 5);
    assert(*commonIter2 == 1);
    std::ranges::iter_swap(commonIter2, commonIter1);
  }
  {
    auto iter1 = forward_iterator<int*>(buffer);
    auto commonIter1 = std::counted_iterator<decltype(iter1)>(iter1, 8);
    auto commonIter2 = std::counted_iterator<decltype(iter1)>(iter1, 8);
    for (auto i = 0; i < 4; ++i) ++commonIter2;
    assert(*commonIter2 == 5);
    std::ranges::iter_swap(commonIter1, commonIter2);
    assert(*commonIter1 == 5);
    assert(*commonIter2 == 1);
    std::ranges::iter_swap(commonIter2, commonIter1);
  }
  {
    auto iter1 = random_access_iterator<int*>(buffer);
    auto commonIter1 = std::counted_iterator<decltype(iter1)>(iter1, 8);
    auto commonIter2 = std::counted_iterator<decltype(iter1)>(iter1, 8);
    for (auto i = 0; i < 4; ++i) ++commonIter2;
    assert(*commonIter2 == 5);
    std::ranges::iter_swap(commonIter1, commonIter2);
    assert(*commonIter1 == 5);
    assert(*commonIter2 == 1);
    std::ranges::iter_swap(commonIter2, commonIter1);
  }

  // Test noexceptness.
  {
    static_assert( noexcept(std::ranges::iter_swap(
      std::declval<std::counted_iterator<HasNoexceptIterSwap<true>>&>(),
      std::declval<std::counted_iterator<HasNoexceptIterSwap<true>>&>()
    )));
    static_assert(!noexcept(std::ranges::iter_swap(
      std::declval<std::counted_iterator<HasNoexceptIterSwap<false>>&>(),
      std::declval<std::counted_iterator<HasNoexceptIterSwap<false>>&>()
    )));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
