//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <class U>
//   requires HasAssign<Iter, const U&>
//   move_iterator&
//   operator=(const move_iterator<U>& u);

#include <iterator>
#include <cassert>

#include "../../../../iterators.h"

template <class It, class U>
void
test(U u)
{
    const std::move_iterator<U> r2(u);
    std::move_iterator<It> r1;
    std::move_iterator<It>& rr = r1 = r2;
    assert(r1.base() == u);
    assert(&rr == &r1);
}

struct base {};
struct derived : base {};

int main()
{
    derived d;

    test<input_iterator<base*> >(input_iterator<derived*>(&d));
    test<forward_iterator<base*> >(forward_iterator<derived*>(&d));
    test<bidirectional_iterator<base*> >(bidirectional_iterator<derived*>(&d));
    test<random_access_iterator<const base*> >(random_access_iterator<derived*>(&d));
    test<base*>(&d);
}
