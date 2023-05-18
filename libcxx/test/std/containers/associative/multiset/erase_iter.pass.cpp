//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// iterator erase(const_iterator position);

#include <set>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

struct TemplateConstructor
{
    template<typename T>
    TemplateConstructor (const T&) {}
};

bool operator<(const TemplateConstructor&, const TemplateConstructor&) { return false; }

int main(int, char**)
{
    {
        typedef std::multiset<int> M;
        typedef int V;
        typedef M::iterator I;
        V ar[] =
        {
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8
        };
        M m(ar, ar + sizeof(ar)/sizeof(ar[0]));
        assert(m.size() == 8);
        I i = m.erase(std::next(m.cbegin(), 3));
        assert(m.size() == 7);
        assert(i == std::next(m.begin(), 3));
        assert(*std::next(m.begin(), 0) == 1);
        assert(*std::next(m.begin(), 1) == 2);
        assert(*std::next(m.begin(), 2) == 3);
        assert(*std::next(m.begin(), 3) == 5);
        assert(*std::next(m.begin(), 4) == 6);
        assert(*std::next(m.begin(), 5) == 7);
        assert(*std::next(m.begin(), 6) == 8);

        i = m.erase(std::next(m.cbegin(), 0));
        assert(m.size() == 6);
        assert(i == m.begin());
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 3);
        assert(*std::next(m.begin(), 2) == 5);
        assert(*std::next(m.begin(), 3) == 6);
        assert(*std::next(m.begin(), 4) == 7);
        assert(*std::next(m.begin(), 5) == 8);

        i = m.erase(std::next(m.cbegin(), 5));
        assert(m.size() == 5);
        assert(i == m.end());
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 3);
        assert(*std::next(m.begin(), 2) == 5);
        assert(*std::next(m.begin(), 3) == 6);
        assert(*std::next(m.begin(), 4) == 7);

        i = m.erase(std::next(m.cbegin(), 1));
        assert(m.size() == 4);
        assert(i == std::next(m.begin()));
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 5);
        assert(*std::next(m.begin(), 2) == 6);
        assert(*std::next(m.begin(), 3) == 7);

        i = m.erase(std::next(m.cbegin(), 2));
        assert(m.size() == 3);
        assert(i == std::next(m.begin(), 2));
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 5);
        assert(*std::next(m.begin(), 2) == 7);

        i = m.erase(std::next(m.cbegin(), 2));
        assert(m.size() == 2);
        assert(i == std::next(m.begin(), 2));
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 5);

        i = m.erase(std::next(m.cbegin(), 0));
        assert(m.size() == 1);
        assert(i == std::next(m.begin(), 0));
        assert(*std::next(m.begin(), 0) == 5);

        i = m.erase(m.cbegin());
        assert(m.size() == 0);
        assert(i == m.begin());
        assert(i == m.end());
    }
#if TEST_STD_VER >= 11
    {
        typedef std::multiset<int, std::less<int>, min_allocator<int>> M;
        typedef int V;
        typedef M::iterator I;
        V ar[] =
        {
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8
        };
        M m(ar, ar + sizeof(ar)/sizeof(ar[0]));
        assert(m.size() == 8);
        I i = m.erase(std::next(m.cbegin(), 3));
        assert(m.size() == 7);
        assert(i == std::next(m.begin(), 3));
        assert(*std::next(m.begin(), 0) == 1);
        assert(*std::next(m.begin(), 1) == 2);
        assert(*std::next(m.begin(), 2) == 3);
        assert(*std::next(m.begin(), 3) == 5);
        assert(*std::next(m.begin(), 4) == 6);
        assert(*std::next(m.begin(), 5) == 7);
        assert(*std::next(m.begin(), 6) == 8);

        i = m.erase(std::next(m.cbegin(), 0));
        assert(m.size() == 6);
        assert(i == m.begin());
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 3);
        assert(*std::next(m.begin(), 2) == 5);
        assert(*std::next(m.begin(), 3) == 6);
        assert(*std::next(m.begin(), 4) == 7);
        assert(*std::next(m.begin(), 5) == 8);

        i = m.erase(std::next(m.cbegin(), 5));
        assert(m.size() == 5);
        assert(i == m.end());
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 3);
        assert(*std::next(m.begin(), 2) == 5);
        assert(*std::next(m.begin(), 3) == 6);
        assert(*std::next(m.begin(), 4) == 7);

        i = m.erase(std::next(m.cbegin(), 1));
        assert(m.size() == 4);
        assert(i == std::next(m.begin()));
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 5);
        assert(*std::next(m.begin(), 2) == 6);
        assert(*std::next(m.begin(), 3) == 7);

        i = m.erase(std::next(m.cbegin(), 2));
        assert(m.size() == 3);
        assert(i == std::next(m.begin(), 2));
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 5);
        assert(*std::next(m.begin(), 2) == 7);

        i = m.erase(std::next(m.cbegin(), 2));
        assert(m.size() == 2);
        assert(i == std::next(m.begin(), 2));
        assert(*std::next(m.begin(), 0) == 2);
        assert(*std::next(m.begin(), 1) == 5);

        i = m.erase(std::next(m.cbegin(), 0));
        assert(m.size() == 1);
        assert(i == std::next(m.begin(), 0));
        assert(*std::next(m.begin(), 0) == 5);

        i = m.erase(m.cbegin());
        assert(m.size() == 0);
        assert(i == m.begin());
        assert(i == m.end());
    }
#endif
#if TEST_STD_VER >= 14
    {
    //  This is LWG #2059
        typedef TemplateConstructor T;
        typedef std::multiset<T> C;
        typedef C::iterator I;

        C c;
        T a{0};
        I it = c.find(a);
        if (it != c.end())
            c.erase(it);
    }
#endif

  return 0;
}
