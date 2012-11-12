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

// requires RandomAccessIterator<Iter>
//   unspecified operator[](difference_type n) const;

#include <iterator>
#include <cassert>

#include "../../../../iterators.h"

template <class It>
void
test(It i, typename std::iterator_traits<It>::difference_type n,
     typename std::iterator_traits<It>::value_type x)
{
    typedef typename std::iterator_traits<It>::value_type value_type;
    const std::reverse_iterator<It> r(i);
    value_type rr = r[n];
    assert(rr == x);
}

int main()
{
    const char* s = "1234567890";
    test(random_access_iterator<const char*>(s+5), 4, '1');
    test(s+5, 4, '1');
}
