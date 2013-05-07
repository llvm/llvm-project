//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <initializer_list>

// template<class E> const E* begin(initializer_list<E> il);

#include <initializer_list>
#include <cassert>

#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS

struct A
{
    A(std::initializer_list<int> il)
    {
        const int* b = begin(il);
        const int* e = end(il);
        assert(il.size() == 3);
        assert(e - b == il.size());
        assert(*b++ == 3);
        assert(*b++ == 2);
        assert(*b++ == 1);
    }
};

#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    A test1 = {3, 2, 1};
#endif
}
