//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void clear();

#include <forward_list>
#include <cassert>

#include "../../../NotConstructible.h"
#include "min_allocator.h"

int main()
{
    {
        typedef NotConstructible T;
        typedef std::forward_list<T> C;
        C c;
        c.clear();
        assert(distance(c.begin(), c.end()) == 0);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t[] = {0, 1, 2, 3, 4};
        C c(std::begin(t), std::end(t));

        c.clear();
        assert(distance(c.begin(), c.end()) == 0);

        c.clear();
        assert(distance(c.begin(), c.end()) == 0);
    }
#if TEST_STD_VER >= 11
    {
        typedef NotConstructible T;
        typedef std::forward_list<T, min_allocator<T>> C;
        C c;
        c.clear();
        assert(distance(c.begin(), c.end()) == 0);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t[] = {0, 1, 2, 3, 4};
        C c(std::begin(t), std::end(t));

        c.clear();
        assert(distance(c.begin(), c.end()) == 0);

        c.clear();
        assert(distance(c.begin(), c.end()) == 0);
    }
#endif
}
