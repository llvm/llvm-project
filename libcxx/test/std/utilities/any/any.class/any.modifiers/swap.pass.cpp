//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <any>

// any::swap(any &) noexcept

// Test swap(large, small) and swap(small, large)

#include <any>
#include <cassert>

#include "any_helpers.h"

using std::any;
using std::any_cast;

template <class LHS, class RHS>
void test_swap() {
    assert(LHS::count == 0);
    assert(RHS::count == 0);
    {
        any a1((LHS(1)));
        any a2(RHS{2});
        assert(LHS::count == 1);
        assert(RHS::count == 1);

        a1.swap(a2);

        assert(LHS::count == 1);
        assert(RHS::count == 1);

        assertContains<RHS>(a1, 2);
        assertContains<LHS>(a2, 1);
    }
    assert(LHS::count == 0);
    assert(RHS::count == 0);
    assert(LHS::copied == 0);
    assert(RHS::copied == 0);
}

template <class Tp>
void test_swap_empty() {
    assert(Tp::count == 0);
    {
        any a1((Tp(1)));
        any a2;
        assert(Tp::count == 1);

        a1.swap(a2);

        assert(Tp::count == 1);

        assertContains<Tp>(a2, 1);
        assertEmpty(a1);
    }
    assert(Tp::count == 0);
    {
        any a1((Tp(1)));
        any a2;
        assert(Tp::count == 1);

        a2.swap(a1);

        assert(Tp::count == 1);

        assertContains<Tp>(a2, 1);
        assertEmpty(a1);
    }
    assert(Tp::count == 0);
    assert(Tp::copied == 0);
}

void test_noexcept()
{
    any a1;
    any a2;
    static_assert(
        noexcept(a1.swap(a2))
      , "any::swap(any&) must be noexcept"
      );
}

int main()
{
    test_noexcept();
    test_swap_empty<small>();
    test_swap_empty<large>();
    test_swap<small1, small2>();
    test_swap<large1, large2>();
    test_swap<small, large>();
    test_swap<large, small>();
}
