//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// template <class U>
//   requires HasConstructor<Iter, const U&>
//   reverse_iterator(const reverse_iterator<U> &u);

#include <iterator>
#include <cassert>

#include "../../../../iterators.h"

template <class It, class U>
void
test(U u)
{
    const std::reverse_iterator<U> r2(u);
    std::reverse_iterator<It> r1 = r2;
    assert(r1.base() == u);
}

struct base {};
struct derived : base {};

int main()
{
    derived d;

    test<bidirectional_iterator<base*> >(bidirectional_iterator<derived*>(&d));
    test<random_access_iterator<const base*> >(random_access_iterator<derived*>(&d));
    test<base*>(&d);
}
