//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <class U>
//   requires HasAssign<Iter, const U&>
//   move_iterator&
//   operator=(const move_iterator<U>& u);
//
//  constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It, class U>
void
test(U u)
{
    const std::move_iterator<U> r2(u);
    std::move_iterator<It> r1;
    std::move_iterator<It>& rr = (r1 = r2);
    assert(r1.base() == u);
    assert(&rr == &r1);
}

struct Base {};
struct Derived : Base {};

int main(int, char**)
{
    Derived d;

    test<cpp17_input_iterator<Base*> >(cpp17_input_iterator<Derived*>(&d));
    test<forward_iterator<Base*> >(forward_iterator<Derived*>(&d));
    test<bidirectional_iterator<Base*> >(bidirectional_iterator<Derived*>(&d));
    test<random_access_iterator<const Base*> >(random_access_iterator<Derived*>(&d));
    test<Base*>(&d);
#if TEST_STD_VER > 14
    {
    using BaseIter    = std::move_iterator<const Base *>;
    using DerivedIter = std::move_iterator<const Derived *>;
    constexpr const Derived *p = nullptr;
    constexpr DerivedIter     it1 = std::make_move_iterator(p);
    constexpr BaseIter        it2 = (BaseIter{nullptr} = it1);
    static_assert(it2.base() == p, "");
    }
#endif

  return 0;
}
