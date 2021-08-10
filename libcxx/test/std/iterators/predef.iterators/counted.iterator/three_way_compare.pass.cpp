//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// template<common_with<I> I2>
//   friend constexpr strong_ordering operator<=>(
//     const counted_iterator& x, const counted_iterator<I2>& y);

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

// This iterator is common_with forward_iterator but NOT comparable with it.
template <class It>
class CommonWithForwardIter
{
    It it_;

public:
    typedef          std::input_iterator_tag                   iterator_category;
    typedef typename std::iterator_traits<It>::value_type      value_type;
    typedef typename std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename std::iterator_traits<It>::reference       reference;

    constexpr It base() const {return it_;}

    CommonWithForwardIter() = default;
    explicit constexpr CommonWithForwardIter(It it) : it_(it) {}
    constexpr CommonWithForwardIter(const forward_iterator<It>& it) : it_(it.base()) {}

    constexpr reference operator*() const {return *it_;}

    constexpr CommonWithForwardIter& operator++() {++it_; return *this;}
    constexpr CommonWithForwardIter operator++(int)
        {CommonWithForwardIter tmp(*this); ++(*this); return tmp;}
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  auto& Eq = std::strong_ordering::equal;
  auto& Less = std::strong_ordering::less;
  auto& Greater = std::strong_ordering::greater;

  {
    {
      std::counted_iterator iter1(forward_iterator<int*>(buffer), 8);
      std::counted_iterator iter2(CommonWithForwardIter<int*>(buffer), 8);

      assert((iter1 <=> iter2) == Eq);
      assert((iter2 <=> iter1) == Eq);
      ++iter1;
      assert((iter1 <=> iter2) == Greater);
      assert((iter2 <=> iter1) == Less);
    }
    {
      std::counted_iterator iter1(forward_iterator<int*>(buffer), 8);
      std::counted_iterator iter2(forward_iterator<int*>(buffer), 8);

      assert((iter1 <=> iter2) == Eq);
      assert((iter2 <=> iter1) == Eq);
      ++iter1;
      assert((iter1 <=> iter2) == Greater);
      assert((iter2 <=> iter1) == Less);
    }
  }


  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
