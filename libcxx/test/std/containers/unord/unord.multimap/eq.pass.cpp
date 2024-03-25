//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash, class Pred, class Alloc>
// bool
// operator==(const unordered_multimap<Key, T, Hash, Pred, Alloc>& x,
//            const unordered_multimap<Key, T, Hash, Pred, Alloc>& y);
//
// template <class Key, class T, class Hash, class Pred, class Alloc>
// bool
// operator!=(const unordered_multimap<Key, T, Hash, Pred, Alloc>& x,
//            const unordered_multimap<Key, T, Hash, Pred, Alloc>& y);

#include <unordered_map>
#include <string>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "min_allocator.h"

#include "test_comparisons.h"

int main(int, char**)
{
    {
        typedef std::unordered_multimap<int, std::string> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(10, "ten"),
            P(20, "twenty"),
            P(20, "twenty 2"),
            P(30, "thirty"),
            P(40, "forty"),
            P(50, "fifty"),
            P(50, "fifty 2"),
            P(50, "fifty 3"),
            P(60, "sixty"),
            P(70, "seventy"),
            P(80, "eighty"),
        };
        const C c1(std::begin(a), std::end(a));
        const C c2;
        assert(testEquality(c1, c2, false));
    }
    {
        typedef std::unordered_multimap<int, std::string> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(10, "ten"),
            P(20, "twenty"),
            P(20, "twenty 2"),
            P(30, "thirty"),
            P(40, "forty"),
            P(50, "fifty"),
            P(50, "fifty 2"),
            P(50, "fifty 3"),
            P(60, "sixty"),
            P(70, "seventy"),
            P(80, "eighty"),
        };
        const C c1(std::begin(a), std::end(a));
        const C c2 = c1;
        assert(testEquality(c1, c2, true));
    }
    {
        typedef std::unordered_multimap<int, std::string> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(10, "ten"),
            P(20, "twenty"),
            P(20, "twenty 2"),
            P(30, "thirty"),
            P(40, "forty"),
            P(50, "fifty"),
            P(50, "fifty 2"),
            P(50, "fifty 3"),
            P(60, "sixty"),
            P(70, "seventy"),
            P(80, "eighty"),
        };
        C c1(std::begin(a), std::end(a));
        C c2 = c1;
        c2.rehash(30);
        assert(testEquality(c1, c2, true));
        c2.insert(P(90, "ninety"));
        assert(testEquality(c1, c2, false));
        c1.insert(P(90, "ninety"));
        assert(testEquality(c1, c2, true));
    }
    {
        typedef std::unordered_multimap<int, std::string> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(10, "ten"),
            P(20, "twenty"),
            P(20, "twenty 2"),
            P(30, "thirty"),
            P(40, "forty"),
            P(50, "fifty"),
            P(50, "fifty 2"),
            P(50, "fifty 3"),
            P(60, "sixty"),
            P(70, "seventy"),
            P(80, "eighty"),
        };
        C c1(std::begin(a), std::end(a));
        C c2 = c1;
        assert(testEquality(c1, c2, true));
        c1.insert(P(70, "seventy 2"));
        c2.insert(P(80, "eighty 2"));
        assert(testEquality(c1, c2, false));
    }
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_multimap<int, std::string, std::hash<int>, std::equal_to<int>,
                            min_allocator<std::pair<const int, std::string>>> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(10, "ten"),
            P(20, "twenty"),
            P(20, "twenty 2"),
            P(30, "thirty"),
            P(40, "forty"),
            P(50, "fifty"),
            P(50, "fifty 2"),
            P(50, "fifty 3"),
            P(60, "sixty"),
            P(70, "seventy"),
            P(80, "eighty"),
        };
        const C c1(std::begin(a), std::end(a));
        const C c2;
        assert(testEquality(c1, c2, false));
    }
    {
        typedef std::unordered_multimap<int, std::string, std::hash<int>, std::equal_to<int>,
                            min_allocator<std::pair<const int, std::string>>> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(10, "ten"),
            P(20, "twenty"),
            P(20, "twenty 2"),
            P(30, "thirty"),
            P(40, "forty"),
            P(50, "fifty"),
            P(50, "fifty 2"),
            P(50, "fifty 3"),
            P(60, "sixty"),
            P(70, "seventy"),
            P(80, "eighty"),
        };
        const C c1(std::begin(a), std::end(a));
        const C c2 = c1;
        assert(testEquality(c1, c2, true));
    }
    {
        typedef std::unordered_multimap<int, std::string, std::hash<int>, std::equal_to<int>,
                            min_allocator<std::pair<const int, std::string>>> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(10, "ten"),
            P(20, "twenty"),
            P(20, "twenty 2"),
            P(30, "thirty"),
            P(40, "forty"),
            P(50, "fifty"),
            P(50, "fifty 2"),
            P(50, "fifty 3"),
            P(60, "sixty"),
            P(70, "seventy"),
            P(80, "eighty"),
        };
        C c1(std::begin(a), std::end(a));
        C c2 = c1;
        c2.rehash(30);
        assert(testEquality(c1, c2, true));
        c2.insert(P(90, "ninety"));
        assert(testEquality(c1, c2, false));
        c1.insert(P(90, "ninety"));
        assert(testEquality(c1, c2, true));
    }
    {
        typedef std::unordered_multimap<int, std::string, std::hash<int>, std::equal_to<int>,
                            min_allocator<std::pair<const int, std::string>>> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(10, "ten"),
            P(20, "twenty"),
            P(20, "twenty 2"),
            P(30, "thirty"),
            P(40, "forty"),
            P(50, "fifty"),
            P(50, "fifty 2"),
            P(50, "fifty 3"),
            P(60, "sixty"),
            P(70, "seventy"),
            P(80, "eighty"),
        };
        C c1(std::begin(a), std::end(a));
        C c2 = c1;
        assert(testEquality(c1, c2, true));
        c1.insert(P(70, "seventy 2"));
        c2.insert(P(80, "eighty 2"));
        assert(testEquality(c1, c2, false));
    }
#endif

    // Make sure we take into account the number of times that a key repeats into equality.
    {
        typedef std::pair<int, char> P;
        P a[] = {P(1, 'a'), P(1, 'b'),            P(1, 'd'), P(2, 'b')};
        P b[] = {P(1, 'a'), P(1, 'b'), P(1, 'b'), P(1, 'd'), P(2, 'b')};

        std::unordered_multimap<int, char> c1(std::begin(a), std::end(a));
        std::unordered_multimap<int, char> c2(std::begin(b), std::end(b));
        assert(testEquality(c1, c2, false));
    }

    // Make sure we incorporate the values into the equality of the maps.
    // If we were to compare only the keys (including how many time each key repeats),
    // the following test would fail cause only the values differ.
    {
        typedef std::pair<int, char> P;
        P a[] = {P(1, 'a'), P(1, 'b'), P(1, 'd'), P(2, 'b')};
        P b[] = {P(1, 'a'), P(1, 'b'), P(1, 'E'), P(2, 'b')};
        //                                   ^ different here

        std::unordered_multimap<int, char> c1(std::begin(a), std::end(a));
        std::unordered_multimap<int, char> c2(std::begin(b), std::end(b));
        assert(testEquality(c1, c2, false));
    }

    return 0;
}
