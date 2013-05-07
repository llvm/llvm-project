//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// template <class InputIterator>
//   deque(InputIterator f, InputIterator l, const allocator_type& a);

#include <deque>
#include <cassert>

#include "test_iterators.h"
#include "../../../test_allocator.h"

template <class InputIterator, class Allocator>
void
test(InputIterator f, InputIterator l, const Allocator& a)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    typedef std::deque<T, Allocator> C;
    typedef typename C::const_iterator const_iterator;
    C d(f, l, a);
    assert(d.get_allocator() == a);
    assert(d.size() == std::distance(f, l));
    assert(distance(d.begin(), d.end()) == d.size());
    for (const_iterator i = d.begin(), e = d.end(); i != e; ++i, ++f)
        assert(*i == *f);
}

int main()
{
    int ab[] = {3, 4, 2, 8, 0, 1, 44, 34, 45, 96, 80, 1, 13, 31, 45};
    int* an = ab + sizeof(ab)/sizeof(ab[0]);
    test(input_iterator<const int*>(ab), input_iterator<const int*>(an), test_allocator<int>(3));
    test(forward_iterator<const int*>(ab), forward_iterator<const int*>(an), test_allocator<int>(4));
    test(bidirectional_iterator<const int*>(ab), bidirectional_iterator<const int*>(an), test_allocator<int>(5));
    test(random_access_iterator<const int*>(ab), random_access_iterator<const int*>(an), test_allocator<int>(6));
}
