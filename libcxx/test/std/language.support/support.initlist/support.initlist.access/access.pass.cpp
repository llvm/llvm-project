//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// template<class E> class initializer_list;

// const E* begin() const;
// const E* end() const;
// size_t size() const;

#include <initializer_list>
#include <cassert>

#include "test_macros.h"

#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS

struct A
{
    A(std::initializer_list<int> il)
    {
        const int* b = il.begin();
        const int* e = il.end();
        assert(il.size() == 3);
        assert(e - b == il.size());
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
        const int* b = il.begin();
        const int* e = il.end();
        assert(il.size() == 3);
        assert(e - b == il.size());
        assert(*b++ == 3);
        assert(*b++ == 2);
        assert(*b++ == 1);
    }
};

#endif  // TEST_STD_VER > 11

#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    A test1 = {3, 2, 1};
#endif
#if TEST_STD_VER > 11
    constexpr B test2 = {3, 2, 1};
#endif  // TEST_STD_VER > 11
}
