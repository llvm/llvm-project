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

#include <iterator>
#include <sstream>
#include <cassert>

int main()
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
    assert(std::operator!=(i1, i3));
}
