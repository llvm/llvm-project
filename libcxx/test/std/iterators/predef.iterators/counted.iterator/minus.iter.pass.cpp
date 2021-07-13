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
//   friend constexpr iter_difference_t<I2> operator-(
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

  {
    std::counted_iterator iter1(random_access_iterator<int*>{buffer}, 8);
    std::counted_iterator iter2(random_access_iterator<int*>{buffer + 4}, 4);
    assert(iter1 - iter2 == -4);
    assert(iter2 - iter1 == 4);
    assert(iter1.count() == 8);
    assert(iter2.count() == 4);

    ASSERT_SAME_TYPE(decltype(iter1 - iter2), std::iter_difference_t<int*>);
  }
  {
    const std::counted_iterator iter1(random_access_iterator<int*>{buffer}, 8);
    const std::counted_iterator iter2(random_access_iterator<int*>{buffer + 4}, 4);
    assert(iter1 - iter2 == -4);
    assert(iter2 - iter1 == 4);
    assert(iter1.count() == 8);
    assert(iter2.count() == 4);

    ASSERT_SAME_TYPE(decltype(iter1 - iter2), std::iter_difference_t<int*>);
  }
  {
    std::counted_iterator iter1(contiguous_iterator<int*>{buffer}, 8);
    std::counted_iterator iter2(contiguous_iterator<int*>{buffer + 2}, 6);
    assert(iter1 - iter2 == -2);
    assert(iter2 - iter1 == 2);
    assert(iter1.count() == 8);
    assert(iter2.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter1 - iter2), std::iter_difference_t<int*>);
  }
  {
    const std::counted_iterator iter1(contiguous_iterator<int*>{buffer}, 8);
    const std::counted_iterator iter2(contiguous_iterator<int*>{buffer + 2}, 6);
    assert(iter1 - iter2 == -2);
    assert(iter2 - iter1 == 2);
    assert(iter1.count() == 8);
    assert(iter2.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter1 - iter2), std::iter_difference_t<int*>);
  }
  // The minus operator works even if Iter is not a random_access_iterator because
  // counted_iterator is able to implement it by subtracting the counts rather than
  // the underlying iterators.
  {
    std::counted_iterator iter1(CommonWithForwardIter<int*>{buffer}, 8);
    std::counted_iterator iter2(forward_iterator<int*>{buffer + 2}, 6);
    assert(iter1 - iter2 == -2);
    assert(iter2 - iter1 == 2);
    assert(iter1.count() == 8);
    assert(iter2.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter1 - iter2), std::iter_difference_t<int*>);
  }
  {
    const std::counted_iterator iter1(CommonWithForwardIter<int*>{buffer}, 8);
    const std::counted_iterator iter2(forward_iterator<int*>{buffer + 2}, 6);
    assert(iter1 - iter2 == -2);
    assert(iter2 - iter1 == 2);
    assert(iter1.count() == 8);
    assert(iter2.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter1 - iter2), std::iter_difference_t<int*>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
