//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// template <class InputIterator>
//   list(InputIterator first, InputIterator last, const Allocator& = Allocator());

#include <list>
#include <cassert>
#include "test_iterators.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main()
{
    {
        int a[] = {0, 1, 2, 3};
        std::list<int> l(input_iterator<const int*>(a),
                         input_iterator<const int*>(a + sizeof(a)/sizeof(a[0])));
        assert(l.size() == sizeof(a)/sizeof(a[0]));
        assert(std::distance(l.begin(), l.end()) == sizeof(a)/sizeof(a[0]));
        int j = 0;
        for (std::list<int>::const_iterator i = l.begin(), e = l.end(); i != e; ++i, ++j)
            assert(*i == j);
    }
    {
        int a[] = {0, 1, 2, 3};
        std::list<int> l(input_iterator<const int*>(a),
                         input_iterator<const int*>(a + sizeof(a)/sizeof(a[0])),
                         std::allocator<int>());
        assert(l.size() == sizeof(a)/sizeof(a[0]));
        assert(std::distance(l.begin(), l.end()) == sizeof(a)/sizeof(a[0]));
        int j = 0;
        for (std::list<int>::const_iterator i = l.begin(), e = l.end(); i != e; ++i, ++j)
            assert(*i == j);
    }
    {
        int a[] = {0, 1, 2, 3};
        // Add 2 for implementations that dynamically allocate a sentinel node and container proxy.
        std::list<int, limited_allocator<int, sizeof(a)/sizeof(a[0]) + 2> > l(input_iterator<const int*>(a),
                         input_iterator<const int*>(a + sizeof(a)/sizeof(a[0])));
        assert(l.size() == sizeof(a)/sizeof(a[0]));
        assert(std::distance(l.begin(), l.end()) == sizeof(a)/sizeof(a[0]));
        int j = 0;
        for (std::list<int>::const_iterator i = l.begin(), e = l.end(); i != e; ++i, ++j)
            assert(*i == j);
    }
#if TEST_STD_VER >= 11
    {
        int a[] = {0, 1, 2, 3};
        std::list<int, min_allocator<int>> l(input_iterator<const int*>(a),
                         input_iterator<const int*>(a + sizeof(a)/sizeof(a[0])));
        assert(l.size() == sizeof(a)/sizeof(a[0]));
        assert(std::distance(l.begin(), l.end()) == sizeof(a)/sizeof(a[0]));
        int j = 0;
        for (std::list<int, min_allocator<int>>::const_iterator i = l.begin(), e = l.end(); i != e; ++i, ++j)
            assert(*i == j);
    }
    {
        int a[] = {0, 1, 2, 3};
        std::list<int, min_allocator<int>> l(input_iterator<const int*>(a),
                         input_iterator<const int*>(a + sizeof(a)/sizeof(a[0])),
                         min_allocator<int>());
        assert(l.size() == sizeof(a)/sizeof(a[0]));
        assert(std::distance(l.begin(), l.end()) == sizeof(a)/sizeof(a[0]));
        int j = 0;
        for (std::list<int, min_allocator<int>>::const_iterator i = l.begin(), e = l.end(); i != e; ++i, ++j)
            assert(*i == j);
    }
#endif
}
