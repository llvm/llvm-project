//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <initializer_list>

// template<class E> const E* begin(initializer_list<E> il);

#include <initializer_list>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

struct A
{
    A(std::initializer_list<int> il)
    {
        const int* b = begin(il);
        const int* e = end(il);
        assert(il.size() == 3);
        assert(static_cast<std::size_t>(e - b) == il.size());
        assert(*b++ == 3);
        assert(*b++ == 2);
        assert(*b++ == 1);
    }
};

#if TEST_STD_VER > 11
struct B
{
    constexpr B(std::initializer_list<int> il)
    {
        const int* b = begin(il);
        const int* e = end(il);
        assert(il.size() == 3);
        assert(static_cast<std::size_t>(e - b) == il.size());
        assert(*b++ == 3);
        assert(*b++ == 2);
        assert(*b++ == 1);
    }
};

#endif // TEST_STD_VER > 11

int main(int, char**)
{
    A test1 = {3, 2, 1}; (void)test1;
#if TEST_STD_VER > 11
    constexpr B test2 = {3, 2, 1};
    (void)test2;
#endif // TEST_STD_VER > 11

  return 0;
}
