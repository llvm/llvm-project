//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// vector(const Alloc& = Alloc());

#include <vector>
#include <cassert>

#include "../../test_allocator.h"

template <class C>
void
test0()
{
    C c;
    assert(c.__invariants());
    assert(c.empty());
    assert(c.get_allocator() == typename C::allocator_type());
}

template <class C>
void
test1(const typename C::allocator_type& a)
{
    C c(a);
    assert(c.__invariants());
    assert(c.empty());
    assert(c.get_allocator() == a);
}

int main()
{
    {
    test0<std::vector<bool> >();
    test1<std::vector<bool, test_allocator<bool> > >(test_allocator<bool>(3));
    }
}
