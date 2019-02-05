//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// reference       front();
// const_reference front() const;

#include <forward_list>
#include <cassert>
#include <iterator>

#include "min_allocator.h"

int main()
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        C c(std::begin(t), std::end(t));
        assert(c.front() == 0);
        c.front() = 10;
        assert(c.front() == 10);
        assert(*c.begin() == 10);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        const C c(std::begin(t), std::end(t));
        assert(c.front() == 0);
        assert(*c.begin() == 0);
    }
#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        C c(std::begin(t), std::end(t));
        assert(c.front() == 0);
        c.front() = 10;
        assert(c.front() == 10);
        assert(*c.begin() == 10);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        const C c(std::begin(t), std::end(t));
        assert(c.front() == 0);
        assert(*c.begin() == 0);
    }
#endif
}
