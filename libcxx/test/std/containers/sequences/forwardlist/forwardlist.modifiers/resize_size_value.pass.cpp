//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void resize(size_type n, const value_type& v);

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "DefaultOnly.h"
#include "min_allocator.h"

#if TEST_STD_VER >= 11
#include "container_test_types.h"
#endif

int main(int, char**)
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t[] = {0, 1, 2, 3, 4};
        C c(std::begin(t), std::end(t));

        c.resize(3, 10);
        assert(std::distance(c.begin(), c.end()) == 3);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);

        c.resize(6, 10);
        assert(std::distance(c.begin(), c.end()) == 6);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(*next(c.begin(), 3) == 10);
        assert(*next(c.begin(), 4) == 10);
        assert(*next(c.begin(), 5) == 10);

        c.resize(6, 12);
        assert(std::distance(c.begin(), c.end()) == 6);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(*next(c.begin(), 3) == 10);
        assert(*next(c.begin(), 4) == 10);
        assert(*next(c.begin(), 5) == 10);
    }
#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t[] = {0, 1, 2, 3, 4};
        C c(std::begin(t), std::end(t));

        c.resize(3, 10);
        assert(std::distance(c.begin(), c.end()) == 3);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);

        c.resize(6, 10);
        assert(std::distance(c.begin(), c.end()) == 6);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(*next(c.begin(), 3) == 10);
        assert(*next(c.begin(), 4) == 10);
        assert(*next(c.begin(), 5) == 10);

        c.resize(6, 12);
        assert(std::distance(c.begin(), c.end()) == 6);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(*next(c.begin(), 3) == 10);
        assert(*next(c.begin(), 4) == 10);
        assert(*next(c.begin(), 5) == 10);
    }
    {
        // Test that the allocator's construct method is being used to
        // construct the new elements and that it's called exactly N times.
        typedef std::forward_list<int, ContainerTestAllocator<int, int>> Container;
        ConstructController* cc = getConstructController();
        cc->reset();
        {
            Container c;
            cc->expect<int const&>(6);
            c.resize(6, 42);
            assert(!cc->unchecked());
        }
    }
#endif

  return 0;
}
