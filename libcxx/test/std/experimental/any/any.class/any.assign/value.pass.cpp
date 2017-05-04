//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// XFAIL: with_system_cxx_lib=macosx10.12
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9
// XFAIL: with_system_cxx_lib=macosx10.7
// XFAIL: with_system_cxx_lib=macosx10.8

// <experimental/any>

// any& operator=(any const &);

// Test value copy and move assignment.

#include <experimental/any>
#include <cassert>

#include "experimental_any_helpers.h"
#include "count_new.hpp"
#include "test_macros.h"

using std::experimental::any;
using std::experimental::any_cast;

template <class LHS, class RHS>
void test_assign_value() {
    assert(LHS::count == 0);
    assert(RHS::count == 0);
    LHS::reset();
    RHS::reset();
    {
        any lhs(LHS(1));
        any const rhs(RHS(2));

        assert(LHS::count == 1);
        assert(RHS::count == 1);
        assert(RHS::copied == 0);

        lhs = rhs;

        assert(RHS::copied == 1);
        assert(LHS::count == 0);
        assert(RHS::count == 2);

        assertContains<RHS>(lhs, 2);
        assertContains<RHS>(rhs, 2);
    }
    assert(LHS::count == 0);
    assert(RHS::count == 0);
    LHS::reset();
    RHS::reset();
    {
        any lhs(LHS(1));
        any rhs(RHS(2));

        assert(LHS::count == 1);
        assert(RHS::count == 1);
        assert(RHS::moved == 1);

        lhs = std::move(rhs);

        assert(RHS::moved >= 1);
        assert(RHS::copied == 0);
        assert(LHS::count == 0);
        assert(RHS::count == 1);

        assertContains<RHS>(lhs, 2);
        assertEmpty<RHS>(rhs);
    }
    assert(LHS::count == 0);
    assert(RHS::count == 0);
}

template <class RHS>
void test_assign_value_empty() {
    assert(RHS::count == 0);
    RHS::reset();
    {
        any lhs;
        RHS rhs(42);
        assert(RHS::count == 1);
        assert(RHS::copied == 0);

        lhs = rhs;

        assert(RHS::count == 2);
        assert(RHS::copied == 1);
        assert(RHS::moved >= 0);
        assertContains<RHS>(lhs, 42);
    }
    assert(RHS::count == 0);
    RHS::reset();
    {
        any lhs;
        RHS rhs(42);
        assert(RHS::count == 1);
        assert(RHS::moved == 0);

        lhs = std::move(rhs);

        assert(RHS::count == 2);
        assert(RHS::copied == 0);
        assert(RHS::moved >= 1);
        assertContains<RHS>(lhs, 42);
    }
    assert(RHS::count == 0);
    RHS::reset();
}


template <class Tp, bool Move = false>
void test_assign_throws() {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    auto try_throw=
    [](any& lhs, auto&& rhs) {
        try {
            Move ? lhs = std::move(rhs)
                 : lhs = rhs;
            assert(false);
        } catch (my_any_exception const &) {
            // do nothing
        } catch (...) {
            assert(false);
        }
    };
    // const lvalue to empty
    {
        any lhs;
        Tp rhs(1);
        assert(Tp::count == 1);

        try_throw(lhs, rhs);

        assert(Tp::count == 1);
        assertEmpty<Tp>(lhs);
    }
    {
        any lhs((small(2)));
        Tp  rhs(1);
        assert(small::count == 1);
        assert(Tp::count == 1);

        try_throw(lhs, rhs);

        assert(small::count == 1);
        assert(Tp::count == 1);
        assertContains<small>(lhs, 2);
    }
    {
        any lhs((large(2)));
        Tp rhs(1);
        assert(large::count == 1);
        assert(Tp::count == 1);

        try_throw(lhs, rhs);

        assert(large::count == 1);
        assert(Tp::count == 1);
        assertContains<large>(lhs, 2);
    }
#endif
}

int main() {
    test_assign_value<small1, small2>();
    test_assign_value<large1, large2>();
    test_assign_value<small, large>();
    test_assign_value<large, small>();
    test_assign_value_empty<small>();
    test_assign_value_empty<large>();
    test_assign_throws<small_throws_on_copy>();
    test_assign_throws<large_throws_on_copy>();
    test_assign_throws<throws_on_move, /* Move = */ true>();
}
