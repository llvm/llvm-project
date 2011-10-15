//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <InputIterator Iter>
//   Iter next(Iter x, Iter::difference_type n = 1);

#include <iterator>
#include <cassert>

#include "../../iterators.h"

template <class It>
void
test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    assert(std::next(i, n) == x);
}

template <class It>
void
test(It i, It x)
{
    assert(std::next(i) == x);
}

int main()
{
    const char* s = "1234567890";
    test(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s+10));
    test(bidirectional_iterator<const char*>(s), 10, bidirectional_iterator<const char*>(s+10));
    test(random_access_iterator<const char*>(s), 10, random_access_iterator<const char*>(s+10));
    test(s, 10, s+10);

    test(forward_iterator<const char*>(s), forward_iterator<const char*>(s+1));
    test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s+1));
    test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s+1));
    test(s, s+1);
}
