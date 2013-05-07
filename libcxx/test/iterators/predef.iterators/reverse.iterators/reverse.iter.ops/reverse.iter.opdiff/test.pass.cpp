//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// template <RandomAccessIterator Iter1, RandomAccessIterator Iter2>
//   requires HasMinus<Iter2, Iter1>
//   auto operator-(const reverse_iterator<Iter1>& x, const reverse_iterator<Iter2>& y)
//   -> decltype(y.base() - x.base());

#include <iterator>
#include <cstddef>
#include <cassert>

#include "test_iterators.h"

template <class It1, class It2>
void
test(It1 l, It2 r, std::ptrdiff_t x)
{
    const std::reverse_iterator<It1> r1(l);
    const std::reverse_iterator<It2> r2(r);
    assert((r1 - r2) == x);
}

int main()
{
    char s[3] = {0};
    test(random_access_iterator<const char*>(s), random_access_iterator<char*>(s), 0);
    test(random_access_iterator<char*>(s), random_access_iterator<const char*>(s+1), 1);
    test(random_access_iterator<const char*>(s+1), random_access_iterator<char*>(s), -1);
    test(s, s, 0);
    test(s, s+1, 1);
    test(s+1, s, -1);
}
