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

// explicit vector(size_type n);

#include <vector>
#include <cassert>

template <class C>
void
test(typename C::size_type n)
{
    C c(n);
    assert(c.__invariants());
    assert(c.size() == n);
    assert(c.get_allocator() == typename C::allocator_type());
    for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e; ++i)
        assert(*i == typename C::value_type());
}

int main()
{
    test<std::vector<bool> >(50);
}
