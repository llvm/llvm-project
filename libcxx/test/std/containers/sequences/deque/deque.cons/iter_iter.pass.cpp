//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// template <class InputIterator> deque(InputIterator f, InputIterator l);

#include <deque>
#include <cassert>

#include "test_allocator.h"
#include "test_iterators.h"
#include "min_allocator.h"

template <class InputIterator>
void
test(InputIterator f, InputIterator l)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    typedef std::allocator<T> Allocator;
    typedef std::deque<T, Allocator> C;
    typedef typename C::const_iterator const_iterator;
    C d(f, l);
    assert(d.size() == std::distance(f, l));
    assert(distance(d.begin(), d.end()) == d.size());
    for (const_iterator i = d.begin(), e = d.end(); i != e; ++i, ++f)
        assert(*i == *f);
}

template <class Allocator, class InputIterator>
void
test(InputIterator f, InputIterator l)
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    typedef std::deque<T, Allocator> C;
    typedef typename C::const_iterator const_iterator;
    C d(f, l);
    assert(d.size() == std::distance(f, l));
    assert(distance(d.begin(), d.end()) == d.size());
    for (const_iterator i = d.begin(), e = d.end(); i != e; ++i, ++f)
        assert(*i == *f);
}

int main()
{
    int ab[] = {3, 4, 2, 8, 0, 1, 44, 34, 45, 96, 80, 1, 13, 31, 45};
    int* an = ab + sizeof(ab)/sizeof(ab[0]);
    test(input_iterator<const int*>(ab), input_iterator<const int*>(an));
    test(forward_iterator<const int*>(ab), forward_iterator<const int*>(an));
    test(bidirectional_iterator<const int*>(ab), bidirectional_iterator<const int*>(an));
    test(random_access_iterator<const int*>(ab), random_access_iterator<const int*>(an));
    test<limited_allocator<int, 4096> >(ab, an);
#if TEST_STD_VER >= 11
    test<min_allocator<int> >(ab, an);
#endif
}
