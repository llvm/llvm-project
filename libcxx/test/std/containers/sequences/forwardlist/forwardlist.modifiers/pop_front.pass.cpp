//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void pop_front();

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        typedef std::forward_list<T> C;
        C c;
        c.push_front(1);
        c.push_front(3);
        c.pop_front();
        assert(std::distance(c.begin(), c.end()) == 1);
        assert(c.front() == 1);
        c.pop_front();
        assert(std::distance(c.begin(), c.end()) == 0);
    }
#if TEST_STD_VER >= 11
    {
        typedef MoveOnly T;
        typedef std::forward_list<T> C;
        C c;
        c.push_front(1);
        c.push_front(3);
        c.pop_front();
        assert(std::distance(c.begin(), c.end()) == 1);
        assert(c.front() == 1);
        c.pop_front();
        assert(std::distance(c.begin(), c.end()) == 0);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        typedef std::forward_list<T, min_allocator<T>> C;
        C c;
        c.push_front(1);
        c.push_front(3);
        c.pop_front();
        assert(std::distance(c.begin(), c.end()) == 1);
        assert(c.front() == 1);
        c.pop_front();
        assert(std::distance(c.begin(), c.end()) == 0);
    }
    {
        typedef MoveOnly T;
        typedef std::forward_list<T, min_allocator<T>> C;
        C c;
        c.push_front(1);
        c.push_front(3);
        c.pop_front();
        assert(std::distance(c.begin(), c.end()) == 1);
        assert(c.front() == 1);
        c.pop_front();
        assert(std::distance(c.begin(), c.end()) == 0);
    }
#endif

  return 0;
}
