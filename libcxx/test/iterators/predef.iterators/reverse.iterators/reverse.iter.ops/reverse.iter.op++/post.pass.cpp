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

// reverse_iterator operator++(int);

#include <iterator>
#include <cassert>

#include "../../../../iterators.h"

template <class It>
void
test(It i, It x)
{
    std::reverse_iterator<It> r(i);
    std::reverse_iterator<It> rr = r++;
    assert(r.base() == x);
    assert(rr.base() == i);
}

int main()
{
    const char* s = "123";
    test(bidirectional_iterator<const char*>(s+1), bidirectional_iterator<const char*>(s));
    test(random_access_iterator<const char*>(s+1), random_access_iterator<const char*>(s));
    test(s+1, s);
}
