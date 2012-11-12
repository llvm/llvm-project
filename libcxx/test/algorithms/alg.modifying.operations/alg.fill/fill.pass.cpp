//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, class T>
//   requires OutputIterator<Iter, const T&>
//   void
//   fill(Iter first, Iter last, const T& value);

#include <algorithm>
#include <cassert>

#include "../../iterators.h"

template <class Iter>
void
test_char()
{
    const unsigned n = 4;
    char ca[n] = {0};
    std::fill(Iter(ca), Iter(ca+n), char(1));
    assert(ca[0] == 1);
    assert(ca[1] == 1);
    assert(ca[2] == 1);
    assert(ca[3] == 1);
}

template <class Iter>
void
test_int()
{
    const unsigned n = 4;
    int ia[n] = {0};
    std::fill(Iter(ia), Iter(ia+n), 1);
    assert(ia[0] == 1);
    assert(ia[1] == 1);
    assert(ia[2] == 1);
    assert(ia[3] == 1);
}

int main()
{
    test_char<forward_iterator<char*> >();
    test_char<bidirectional_iterator<char*> >();
    test_char<random_access_iterator<char*> >();
    test_char<char*>();

    test_int<forward_iterator<int*> >();
    test_int<bidirectional_iterator<int*> >();
    test_int<random_access_iterator<int*> >();
    test_int<int*>();
}
