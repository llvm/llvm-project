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

// template <InputIterator Iter1, InputIterator Iter2>
//   requires HasEqualTo<Iter1, Iter2>
//   bool
//   operator==(const move_iterator<Iter1>& x, const move_iterator<Iter2>& y);

#include <iterator>
#include <cassert>

#include "../../../../iterators.h"

template <class It>
void
test(It l, It r, bool x)
{
    const std::move_iterator<It> r1(l);
    const std::move_iterator<It> r2(r);
    assert((r1 == r2) == x);
}

int main()
{
    char s[] = "1234567890";
    test(input_iterator<char*>(s), input_iterator<char*>(s), true);
    test(input_iterator<char*>(s), input_iterator<char*>(s+1), false);
    test(forward_iterator<char*>(s), forward_iterator<char*>(s), true);
    test(forward_iterator<char*>(s), forward_iterator<char*>(s+1), false);
    test(bidirectional_iterator<char*>(s), bidirectional_iterator<char*>(s), true);
    test(bidirectional_iterator<char*>(s), bidirectional_iterator<char*>(s+1), false);
    test(random_access_iterator<char*>(s), random_access_iterator<char*>(s), true);
    test(random_access_iterator<char*>(s), random_access_iterator<char*>(s+1), false);
    test(s, s, true);
    test(s, s+1, false);
}
