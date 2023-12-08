//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <forward_list>

// iterator insert_after(const_iterator p, initializer_list<value_type> il);

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
        I i = c.insert_after(c.cbefore_begin(), {});
        assert(i == c.before_begin());
        assert(std::distance(c.begin(), c.end()) == 0);

        i = c.insert_after(c.cbefore_begin(), {0, 1, 2});
        assert(i == std::next(c.before_begin(), 3));
        assert(std::distance(c.begin(), c.end()) == 3);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 2);

        i = c.insert_after(c.begin(), {3, 4});
        assert(i == std::next(c.begin(), 2));
        assert(std::distance(c.begin(), c.end()) == 5);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 3);
        assert(*std::next(c.begin(), 2) == 4);
        assert(*std::next(c.begin(), 3) == 1);
        assert(*std::next(c.begin(), 4) == 2);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        typedef C::iterator I;
        C c;
        I i = c.insert_after(c.cbefore_begin(), {});
        assert(i == c.before_begin());
        assert(std::distance(c.begin(), c.end()) == 0);

        i = c.insert_after(c.cbefore_begin(), {0, 1, 2});
        assert(i == std::next(c.before_begin(), 3));
        assert(std::distance(c.begin(), c.end()) == 3);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 2);

        i = c.insert_after(c.begin(), {3, 4});
        assert(i == std::next(c.begin(), 2));
        assert(std::distance(c.begin(), c.end()) == 5);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 3);
        assert(*std::next(c.begin(), 2) == 4);
        assert(*std::next(c.begin(), 3) == 1);
        assert(*std::next(c.begin(), 4) == 2);
    }

  return 0;
}
