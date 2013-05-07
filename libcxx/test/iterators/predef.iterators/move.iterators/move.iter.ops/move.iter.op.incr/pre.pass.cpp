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

// move_iterator& operator++();

#include <iterator>
#include <cassert>

#include "test_iterators.h"

template <class It>
void
test(It i, It x)
{
    std::move_iterator<It> r(i);
    std::move_iterator<It>& rr = ++r;
    assert(r.base() == x);
    assert(&rr == &r);
}

int main()
{
    char s[] = "123";
    test(input_iterator<char*>(s), input_iterator<char*>(s+1));
    test(forward_iterator<char*>(s), forward_iterator<char*>(s+1));
    test(bidirectional_iterator<char*>(s), bidirectional_iterator<char*>(s+1));
    test(random_access_iterator<char*>(s), random_access_iterator<char*>(s+1));
    test(s, s+1);
}
