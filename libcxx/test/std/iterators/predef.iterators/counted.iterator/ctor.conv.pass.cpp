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

// template<class I2>
//   requires convertible_to<const I2&, I>
//     constexpr counted_iterator(const counted_iterator<I2>& x);

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

template<class T>
class ConvertibleTo
{
    int *it_;

public:
    typedef          std::input_iterator_tag                      iterator_category;
    typedef int                                                   value_type;
    typedef typename std::iterator_traits<int *>::difference_type difference_type;
    typedef int *                                                 pointer;
    typedef int &                                                 reference;

    constexpr int *base() const {return it_;}

    ConvertibleTo() = default;
    explicit constexpr ConvertibleTo(int *it) : it_(it) {}

    constexpr reference operator*() const {return *it_;}

    constexpr ConvertibleTo& operator++() {++it_; return *this;}
    constexpr ConvertibleTo operator++(int)
        {ConvertibleTo tmp(*this); ++(*this); return tmp;}

    constexpr operator T() const { return T(it_); }
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert( std::is_constructible_v<std::counted_iterator<forward_iterator<int*>>,
                                           std::counted_iterator<forward_iterator<int*>>>);
    static_assert(!std::is_constructible_v<std::counted_iterator<forward_iterator<int*>>,
                                           std::counted_iterator<random_access_iterator<int*>>>);
  }
  {
    std::counted_iterator iter1(ConvertibleTo<forward_iterator<int*>>{buffer}, 8);
    std::counted_iterator<forward_iterator<int*>> iter2(iter1);
    assert(iter2.base() == forward_iterator<int*>{buffer});
    assert(iter2.count() == 8);
  }
  {
    const std::counted_iterator iter1(ConvertibleTo<forward_iterator<int*>>{buffer}, 8);
    const std::counted_iterator<forward_iterator<int*>> iter2(iter1);
    assert(iter2.base() == forward_iterator<int*>{buffer});
    assert(iter2.count() == 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
