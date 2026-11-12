//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// template <class T, class charT, class traits, class Distance>
//   bool operator==(const istream_iterator<T,charT,traits,Distance> &x,
//                   const istream_iterator<T,charT,traits,Distance> &y);
//
// template <class T, class charT, class traits, class Distance>
//   bool operator!=(const istream_iterator<T,charT,traits,Distance> &x,
//                   const istream_iterator<T,charT,traits,Distance> &y);
//
// friend bool operator==(const istream_iterator& i, default_sentinel_t); // since C++20

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  {
    std::istringstream inf1(" 1 23");
    std::istringstream inf2(" 1 23");
    std::istream_iterator<int> i1(inf1);
    std::istream_iterator<int> i2(inf1);
    std::istream_iterator<int> i3(inf2);
    std::istream_iterator<int> i4;
    std::istream_iterator<int> i5;
    assert(i1 == i1);
    assert(i1 == i2);
    assert(i1 != i3);
    assert(i1 != i4);
    assert(i1 != i5);

    assert(i2 == i2);
    assert(i2 != i3);
    assert(i2 != i4);
    assert(i2 != i5);

    assert(i3 == i3);
    assert(i3 != i4);
    assert(i3 != i5);

    assert(i4 == i4);
    assert(i4 == i5);

    assert(std::operator==(i1, i2));
#if TEST_STD_VER <= 17
    assert(std::operator!=(i1, i3));
#endif
  }

#if TEST_STD_VER > 17
  {
    std::istream_iterator<int> i1;
    std::istream_iterator<int> i2(std::default_sentinel);
    assert(i1 == i2);

    assert(i1 == std::default_sentinel);
    assert(i2 == std::default_sentinel);
    assert(std::default_sentinel == i1);
    assert(std::default_sentinel == i2);
    assert(!(i1 != std::default_sentinel));
    assert(!(i2 != std::default_sentinel));
    assert(!(std::default_sentinel != i1));
    assert(!(std::default_sentinel != i2));

    std::istringstream stream(" 1 23");
    std::istream_iterator<int> i3(stream);

    assert(!(i3 == std::default_sentinel));
    assert(!(std::default_sentinel == i3));
    assert(i3 != std::default_sentinel);
    assert(std::default_sentinel != i3);
  }
#endif

  return 0;
}
