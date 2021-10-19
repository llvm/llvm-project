//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// template <class T, class Alloc>
//   void swap(list<T,Alloc>& x, list<T,Alloc>& y);

#include <list>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        int a1[] = {1, 3, 7, 9, 10};
        int a2[] = {0, 2, 4, 5, 6, 8, 11};
        std::list<int> c1(a1, a1 + 5);
        std::list<int> c2(a2, a2 + 7);
        std::list<int>::iterator it1 = c1.begin();
        std::list<int>::const_iterator it2 = c2.begin();
        swap(c1, c2);
        assert(c1 == std::list<int>(a2, a2 + 7));
        assert(c2 == std::list<int>(a1, a1 + 5));
        assert(it1 == c2.begin()); // Iterators remain valid
        assert(it2 == c1.begin()); // Iterators remain valid
    }
    {
        int a2[] = {0, 2, 4, 5, 6, 8, 11};
        std::list<int> c1;
        std::list<int> c2(a2, a2 + 7);
        swap(c1, c2);
        assert(c1 == std::list<int>(a2, a2 + 7));
        assert(c2.empty());
        assert(distance(c2.begin(), c2.end()) == 0);
    }
    {
        int a1[] = {1, 3, 7, 9, 10};
        std::list<int> c1(a1, a1 + 5);
        std::list<int> c2;
        swap(c1, c2);
        assert(c1.empty());
        assert(distance(c1.begin(), c1.end()) == 0);
        assert(c2 == std::list<int>(a1, a1 + 5));
    }
    {
        std::list<int> c1;
        std::list<int> c2;
        swap(c1, c2);
        assert(c1.empty());
        assert(distance(c1.begin(), c1.end()) == 0);
        assert(c2.empty());
        assert(distance(c2.begin(), c2.end()) == 0);
    }
    {
        int a1[] = {1, 3, 7, 9, 10};
        int a2[] = {0, 2, 4, 5, 6, 8, 11};
        typedef test_allocator<int> A;
        std::list<int, A> c1(a1, a1 + 5, A(1));
        std::list<int, A> c2(a2, a2 + 7, A(1));
        swap(c1, c2);
        assert((c1 == std::list<int, A>(a2, a2 + 7)));
        assert(c1.get_allocator() == A(1));
        assert((c2 == std::list<int, A>(a1, a1 + 5)));
        assert(c2.get_allocator() == A(1));
    }
    {
        int a1[] = {1, 3, 7, 9, 10};
        int a2[] = {0, 2, 4, 5, 6, 8, 11};
        typedef other_allocator<int> A;
        std::list<int, A> c1(a1, a1 + 5, A(1));
        std::list<int, A> c2(a2, a2 + 7, A(2));
        swap(c1, c2);
        assert((c1 == std::list<int, A>(a2, a2 + 7)));
        assert(c1.get_allocator() == A(2));
        assert((c2 == std::list<int, A>(a1, a1 + 5)));
        assert(c2.get_allocator() == A(1));
    }
#if TEST_STD_VER >= 11
    {
        int a1[] = {1, 3, 7, 9, 10};
        int a2[] = {0, 2, 4, 5, 6, 8, 11};
        std::list<int, min_allocator<int>> c1(a1, a1 + 5);
        std::list<int, min_allocator<int>> c2(a2, a2 + 7);
        swap(c1, c2);
        assert((c1 == std::list<int, min_allocator<int>>(a2, a2 + 7)));
        assert((c2 == std::list<int, min_allocator<int>>(a1, a1 + 5)));
    }
    {
        int a2[] = {0, 2, 4, 5, 6, 8, 11};
        std::list<int, min_allocator<int>> c1;
        std::list<int, min_allocator<int>> c2(a2, a2 + 7);
        swap(c1, c2);
        assert((c1 == std::list<int, min_allocator<int>>(a2, a2 + 7)));
        assert(c2.empty());
        assert(distance(c2.begin(), c2.end()) == 0);
    }
    {
        int a1[] = {1, 3, 7, 9, 10};
        std::list<int, min_allocator<int>> c1(a1, a1 + 5);
        std::list<int, min_allocator<int>> c2;
        swap(c1, c2);
        assert(c1.empty());
        assert(distance(c1.begin(), c1.end()) == 0);
        assert((c2 == std::list<int, min_allocator<int>>(a1, a1 + 5)));
    }
    {
        std::list<int, min_allocator<int>> c1;
        std::list<int, min_allocator<int>> c2;
        swap(c1, c2);
        assert(c1.empty());
        assert(distance(c1.begin(), c1.end()) == 0);
        assert(c2.empty());
        assert(distance(c2.begin(), c2.end()) == 0);
    }
    {
        int a1[] = {1, 3, 7, 9, 10};
        int a2[] = {0, 2, 4, 5, 6, 8, 11};
        typedef min_allocator<int> A;
        std::list<int, A> c1(a1, a1 + 5, A());
        std::list<int, A> c2(a2, a2 + 7, A());
        swap(c1, c2);
        assert((c1 == std::list<int, A>(a2, a2 + 7)));
        assert(c1.get_allocator() == A());
        assert((c2 == std::list<int, A>(a1, a1 + 5)));
        assert(c2.get_allocator() == A());
    }
#endif

    return 0;
}
