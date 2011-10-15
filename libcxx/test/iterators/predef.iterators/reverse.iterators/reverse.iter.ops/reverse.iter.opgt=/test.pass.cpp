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
//   requires HasGreater<Iter1, Iter2>
//   bool
//   operator>=(const reverse_iterator<Iter1>& x, const reverse_iterator<Iter2>& y);

#include <iterator>
#include <cassert>

#include "../../../../iterators.h"

template <class It>
void
test(It l, It r, bool x)
{
    const std::reverse_iterator<It> r1(l);
    const std::reverse_iterator<It> r2(r);
    assert((r1 >= r2) == x);
}

int main()
{
    const char* s = "1234567890";
    test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s), true);
    test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s+1), true);
    test(random_access_iterator<const char*>(s+1), random_access_iterator<const char*>(s), false);
    test(s, s, true);
    test(s, s+1, true);
    test(s+1, s, false);
}
