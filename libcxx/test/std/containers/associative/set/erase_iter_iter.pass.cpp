//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// iterator erase(const_iterator first, const_iterator last);

#include <set>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::set<int> M;
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
        I i = m.erase(std::next(m.cbegin(), 5), std::next(m.cbegin(), 5));
        assert(m.size() == 8);
        assert(i == std::next(m.begin(), 5));
        assert(*std::next(m.begin(), 0) == 1);
        assert(*std::next(m.begin(), 1) == 2);
        assert(*std::next(m.begin(), 2) == 3);
        assert(*std::next(m.begin(), 3) == 4);
        assert(*std::next(m.begin(), 4) == 5);
        assert(*std::next(m.begin(), 5) == 6);
        assert(*std::next(m.begin(), 6) == 7);
        assert(*std::next(m.begin(), 7) == 8);

        i = m.erase(std::next(m.cbegin(), 3), std::next(m.cbegin(), 4));
        assert(m.size() == 7);
        assert(i == std::next(m.begin(), 3));
        assert(*std::next(m.begin(), 0) == 1);
        assert(*std::next(m.begin(), 1) == 2);
        assert(*std::next(m.begin(), 2) == 3);
        assert(*std::next(m.begin(), 3) == 5);
        assert(*std::next(m.begin(), 4) == 6);
        assert(*std::next(m.begin(), 5) == 7);
        assert(*std::next(m.begin(), 6) == 8);

        i = m.erase(std::next(m.cbegin(), 2), std::next(m.cbegin(), 5));
        assert(m.size() == 4);
        assert(i == std::next(m.begin(), 2));
        assert(*std::next(m.begin(), 0) == 1);
        assert(*std::next(m.begin(), 1) == 2);
        assert(*std::next(m.begin(), 2) == 7);
        assert(*std::next(m.begin(), 3) == 8);

        i = m.erase(std::next(m.cbegin(), 0), std::next(m.cbegin(), 2));
        assert(m.size() == 2);
        assert(i == std::next(m.begin(), 0));
        assert(*std::next(m.begin(), 0) == 7);
        assert(*std::next(m.begin(), 1) == 8);

        i = m.erase(m.cbegin(), m.cend());
        assert(m.size() == 0);
        assert(i == m.end());
    }
#if TEST_STD_VER >= 11
    {
        typedef std::set<int, std::less<int>, min_allocator<int>> M;
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
        I i = m.erase(std::next(m.cbegin(), 5), std::next(m.cbegin(), 5));
        assert(m.size() == 8);
        assert(i == std::next(m.begin(), 5));
        assert(*std::next(m.begin(), 0) == 1);
        assert(*std::next(m.begin(), 1) == 2);
        assert(*std::next(m.begin(), 2) == 3);
        assert(*std::next(m.begin(), 3) == 4);
        assert(*std::next(m.begin(), 4) == 5);
        assert(*std::next(m.begin(), 5) == 6);
        assert(*std::next(m.begin(), 6) == 7);
        assert(*std::next(m.begin(), 7) == 8);

        i = m.erase(std::next(m.cbegin(), 3), std::next(m.cbegin(), 4));
        assert(m.size() == 7);
        assert(i == std::next(m.begin(), 3));
        assert(*std::next(m.begin(), 0) == 1);
        assert(*std::next(m.begin(), 1) == 2);
        assert(*std::next(m.begin(), 2) == 3);
        assert(*std::next(m.begin(), 3) == 5);
        assert(*std::next(m.begin(), 4) == 6);
        assert(*std::next(m.begin(), 5) == 7);
        assert(*std::next(m.begin(), 6) == 8);

        i = m.erase(std::next(m.cbegin(), 2), std::next(m.cbegin(), 5));
        assert(m.size() == 4);
        assert(i == std::next(m.begin(), 2));
        assert(*std::next(m.begin(), 0) == 1);
        assert(*std::next(m.begin(), 1) == 2);
        assert(*std::next(m.begin(), 2) == 7);
        assert(*std::next(m.begin(), 3) == 8);

        i = m.erase(std::next(m.cbegin(), 0), std::next(m.cbegin(), 2));
        assert(m.size() == 2);
        assert(i == std::next(m.begin(), 0));
        assert(*std::next(m.begin(), 0) == 7);
        assert(*std::next(m.begin(), 1) == 8);

        i = m.erase(m.cbegin(), m.cend());
        assert(m.size() == 0);
        assert(i == m.end());
    }
#endif

  return 0;
}
