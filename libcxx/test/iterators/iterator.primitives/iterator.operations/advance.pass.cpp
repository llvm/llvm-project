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
//   void advance(Iter& i, Iter::difference_type n);
//
// template <BidirectionalIterator Iter>
//   void advance(Iter& i, Iter::difference_type n);
//
// template <RandomAccessIterator Iter>
//   void advance(Iter& i, Iter::difference_type n);

#include <iterator>
#include <cassert>

#include "../../iterators.h"

template <class It>
void
test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    std::advance(i, n);
    assert(i == x);
}

int main()
{
    const char* s = "1234567890";
    test(input_iterator<const char*>(s), 10, input_iterator<const char*>(s+10));
    test(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s+10));
    test(bidirectional_iterator<const char*>(s+5), 5, bidirectional_iterator<const char*>(s+10));
    test(bidirectional_iterator<const char*>(s+5), -5, bidirectional_iterator<const char*>(s));
    test(random_access_iterator<const char*>(s+5), 5, random_access_iterator<const char*>(s+10));
    test(random_access_iterator<const char*>(s+5), -5, random_access_iterator<const char*>(s));
    test(s+5, 5, s+10);
    test(s+5, -5, s);
}
