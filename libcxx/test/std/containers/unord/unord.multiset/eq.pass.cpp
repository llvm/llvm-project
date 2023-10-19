//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Key, class Hash, class Pred, class Alloc>
// bool
// operator==(const unordered_multiset<Key, Hash, Pred, Alloc>& x,
//            const unordered_multiset<Key, Hash, Pred, Alloc>& y);
//
// template <class Key, class Hash, class Pred, class Alloc>
// bool
// operator!=(const unordered_multiset<Key, Hash, Pred, Alloc>& x,
//            const unordered_multiset<Key, Hash, Pred, Alloc>& y);

#include <unordered_set>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "min_allocator.h"

#include "test_comparisons.h"

int main(int, char**)
{
    {
        typedef std::unordered_multiset<int> C;
        typedef int P;
        P a[] =
        {
            P(10),
            P(20),
            P(20),
            P(30),
            P(40),
            P(50),
            P(50),
            P(50),
            P(60),
            P(70),
            P(80)
        };
        const C c1(std::begin(a), std::end(a));
        const C c2;
        assert(!(c1 == c2));
        assert( (c1 != c2));
    }
    {
        typedef std::unordered_multiset<int> C;
        typedef int P;
        P a[] =
        {
            P(10),
            P(20),
            P(20),
            P(30),
            P(40),
            P(50),
            P(50),
            P(50),
            P(60),
            P(70),
            P(80)
        };
        const C c1(std::begin(a), std::end(a));
        const C c2 = c1;
        assert( (c1 == c2));
        assert(!(c1 != c2));
    }
    {
        typedef std::unordered_multiset<int> C;
        typedef int P;
        P a[] =
        {
            P(10),
            P(20),
            P(20),
            P(30),
            P(40),
            P(50),
            P(50),
            P(50),
            P(60),
            P(70),
            P(80)
        };
        C c1(std::begin(a), std::end(a));
        C c2 = c1;
        c2.rehash(30);
        assert( (c1 == c2));
        assert(!(c1 != c2));
        c2.insert(P(90));
        assert(!(c1 == c2));
        assert( (c1 != c2));
        c1.insert(P(90));
        assert( (c1 == c2));
        assert(!(c1 != c2));
    }
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_multiset<int, std::hash<int>,
                                      std::equal_to<int>, min_allocator<int>> C;
        typedef int P;
        P a[] =
        {
            P(10),
            P(20),
            P(20),
            P(30),
            P(40),
            P(50),
            P(50),
            P(50),
            P(60),
            P(70),
            P(80)
        };
        const C c1(std::begin(a), std::end(a));
        const C c2;
        assert(!(c1 == c2));
        assert( (c1 != c2));
    }
    {
        typedef std::unordered_multiset<int, std::hash<int>,
                                      std::equal_to<int>, min_allocator<int>> C;
        typedef int P;
        P a[] =
        {
            P(10),
            P(20),
            P(20),
            P(30),
            P(40),
            P(50),
            P(50),
            P(50),
            P(60),
            P(70),
            P(80)
        };
        const C c1(std::begin(a), std::end(a));
        const C c2 = c1;
        assert( (c1 == c2));
        assert(!(c1 != c2));
    }
    {
        typedef std::unordered_multiset<int, std::hash<int>,
                                      std::equal_to<int>, min_allocator<int>> C;
        typedef int P;
        P a[] =
        {
            P(10),
            P(20),
            P(20),
            P(30),
            P(40),
            P(50),
            P(50),
            P(50),
            P(60),
            P(70),
            P(80)
        };
        C c1(std::begin(a), std::end(a));
        C c2 = c1;
        c2.rehash(30);
        assert( (c1 == c2));
        assert(!(c1 != c2));
        c2.insert(P(90));
        assert(!(c1 == c2));
        assert( (c1 != c2));
        c1.insert(P(90));
        assert( (c1 == c2));
        assert(!(c1 != c2));
    }
#endif

    // Make sure we take into account the number of times that a key repeats into equality.
    {
        int a[] = {1, 1, 1, 2};
        int b[] = {1, 1, 1, 1, 2};

        std::unordered_multiset<int> c1(std::begin(a), std::end(a));
        std::unordered_multiset<int> c2(std::begin(b), std::end(b));
        assert(testEquality(c1, c2, false));
    }

    // Make sure we behave properly when a custom key predicate is provided.
    {
        int a[] = {1, 3};
        int b[] = {1, 1};
        // A very poor hash
        struct HashModuloOddness {
            std::size_t operator()(int x) const { return std::hash<int>()(x % 2); }
        };
        // A very poor hash
        struct CompareModuloOddness {
            bool operator()(int x, int y) const { return (x % 2) == (y % 2); }
        };

        using Set = std::unordered_multiset<int, HashModuloOddness, CompareModuloOddness>;
        Set c1(std::begin(a), std::end(a));
        Set c2(std::begin(b), std::end(b));

        assert(testEquality(c1, c2, false));
    }

    return 0;
}
