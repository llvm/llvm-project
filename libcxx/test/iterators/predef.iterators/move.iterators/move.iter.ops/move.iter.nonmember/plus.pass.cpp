//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <RandomAccessIterator Iter>
//   move_iterator<Iter>
//   operator+(Iter::difference_type n, const move_iterator<Iter>& x);

#include <iterator>
#include <cassert>

#include "test_iterators.h"

template <class It>
void
test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    const std::move_iterator<It> r(i);
    std::move_iterator<It> rr = n + r;
    assert(rr.base() == x);
}

int main()
{
    char s[] = "1234567890";
    test(random_access_iterator<char*>(s+5), 5, random_access_iterator<char*>(s+10));
    test(s+5, 5, s+10);
}
