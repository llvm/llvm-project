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

// template <InputIterator Iter>
//   move_iterator<Iter>
//   make_move_iterator(const Iter& i);

#include <iterator>
#include <cassert>

#include "test_iterators.h"

template <class It>
void
test(It i)
{
    const std::move_iterator<It> r(i);
    assert(std::make_move_iterator(i) == r);
}

int main()
{
    char s[] = "1234567890";
    test(input_iterator<char*>(s+5));
    test(forward_iterator<char*>(s+5));
    test(bidirectional_iterator<char*>(s+5));
    test(random_access_iterator<char*>(s+5));
    test(s+5);
}
