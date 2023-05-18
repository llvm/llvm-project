//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// iterator insert_after(const_iterator p, size_type n, const value_type& v);

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        typedef C::iterator I;
        C c;
        I i = c.insert_after(c.cbefore_begin(), 0, 0);
        assert(i == c.before_begin());
        assert(std::distance(c.begin(), c.end()) == 0);

        i = c.insert_after(c.cbefore_begin(), 3, 3);
        assert(i == std::next(c.before_begin(), 3));
        assert(std::distance(c.begin(), c.end()) == 3);
        assert(*std::next(c.begin(), 0) == 3);
        assert(*std::next(c.begin(), 1) == 3);
        assert(*std::next(c.begin(), 2) == 3);

        i = c.insert_after(c.begin(), 2, 2);
        assert(i == std::next(c.begin(), 2));
        assert(std::distance(c.begin(), c.end()) == 5);
        assert(*std::next(c.begin(), 0) == 3);
        assert(*std::next(c.begin(), 1) == 2);
        assert(*std::next(c.begin(), 2) == 2);
        assert(*std::next(c.begin(), 3) == 3);
        assert(*std::next(c.begin(), 4) == 3);
    }
#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        typedef C::iterator I;
        C c;
        I i = c.insert_after(c.cbefore_begin(), 0, 0);
        assert(i == c.before_begin());
        assert(std::distance(c.begin(), c.end()) == 0);

        i = c.insert_after(c.cbefore_begin(), 3, 3);
        assert(i == std::next(c.before_begin(), 3));
        assert(std::distance(c.begin(), c.end()) == 3);
        assert(*std::next(c.begin(), 0) == 3);
        assert(*std::next(c.begin(), 1) == 3);
        assert(*std::next(c.begin(), 2) == 3);

        i = c.insert_after(c.begin(), 2, 2);
        assert(i == std::next(c.begin(), 2));
        assert(std::distance(c.begin(), c.end()) == 5);
        assert(*std::next(c.begin(), 0) == 3);
        assert(*std::next(c.begin(), 1) == 2);
        assert(*std::next(c.begin(), 2) == 2);
        assert(*std::next(c.begin(), 3) == 3);
        assert(*std::next(c.begin(), 4) == 3);
    }
#endif

  return 0;
}
