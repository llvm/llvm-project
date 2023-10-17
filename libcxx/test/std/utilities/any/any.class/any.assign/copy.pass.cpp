//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// XFAIL: availability-bad_any_cast-missing && !no-exceptions

// <any>

// any& operator=(const any&);

// Test copy assignment

#include <any>
#include <cassert>

#include "any_helpers.h"
#include "count_new.h"
#include "test_macros.h"

template <class LHS, class RHS>
void test_copy_assign() {
    assert(LHS::count == 0);
    assert(RHS::count == 0);
    LHS::reset();
    RHS::reset();
    {
        std::any lhs = LHS(1);
        const std::any rhs = RHS(2);

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
}

template <class LHS>
void test_copy_assign_empty() {
    assert(LHS::count == 0);
    LHS::reset();
    {
        std::any lhs;
        const std::any rhs = LHS(42);

        assert(LHS::count == 1);
        assert(LHS::copied == 0);

        lhs = rhs;

        assert(LHS::copied == 1);
        assert(LHS::count == 2);

        assertContains<LHS>(lhs, 42);
        assertContains<LHS>(rhs, 42);
    }
    assert(LHS::count == 0);
    LHS::reset();
    {
        std::any lhs = LHS(1);
        const std::any rhs;

        assert(LHS::count == 1);
        assert(LHS::copied == 0);

        lhs = rhs;

        assert(LHS::copied == 0);
        assert(LHS::count == 0);

        assertEmpty<LHS>(lhs);
        assertEmpty(rhs);
    }
    assert(LHS::count == 0);
}

void test_copy_assign_self() {
    // empty
    {
        std::any a;
        a = (std::any&)a;
        assertEmpty(a);
        assert(globalMemCounter.checkOutstandingNewEq(0));
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    // small
    {
        std::any a = small(1);
        assert(small::count == 1);

        a = (std::any&)a;

        assert(small::count == 1);
        assertContains<small>(a, 1);
        assert(globalMemCounter.checkOutstandingNewEq(0));
    }
    assert(small::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    // large
    {
        std::any a = large(1);
        assert(large::count == 1);

        a = (std::any&)a;

        assert(large::count == 1);
        assertContains<large>(a, 1);
        assert(globalMemCounter.checkOutstandingNewEq(1));
    }
    assert(large::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
}

template <class Tp>
void test_copy_assign_throws()
{
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    auto try_throw =
    [](std::any& lhs, const std::any& rhs) {
        try {
            lhs = rhs;
            assert(false);
        } catch (const my_any_exception&) {
            // do nothing
        } catch (...) {
            assert(false);
        }
    };
    // const lvalue to empty
    {
        std::any lhs;
        const std::any rhs = Tp(1);
        assert(Tp::count == 1);

        try_throw(lhs, rhs);

        assert(Tp::count == 1);
        assertEmpty<Tp>(lhs);
        assertContains<Tp>(rhs, 1);
    }
    {
        std::any lhs = small(2);
        const std::any rhs = Tp(1);
        assert(small::count == 1);
        assert(Tp::count == 1);

        try_throw(lhs, rhs);

        assert(small::count == 1);
        assert(Tp::count == 1);
        assertContains<small>(lhs, 2);
        assertContains<Tp>(rhs, 1);
    }
    {
        std::any lhs = large(2);
        const std::any rhs = Tp(1);
        assert(large::count == 1);
        assert(Tp::count == 1);

        try_throw(lhs, rhs);

        assert(large::count == 1);
        assert(Tp::count == 1);
        assertContains<large>(lhs, 2);
        assertContains<Tp>(rhs, 1);
    }
#endif
}

int main(int, char**) {
    globalMemCounter.reset();
    test_copy_assign<small1, small2>();
    test_copy_assign<large1, large2>();
    test_copy_assign<small, large>();
    test_copy_assign<large, small>();
    test_copy_assign_empty<small>();
    test_copy_assign_empty<large>();
    test_copy_assign_self();
    test_copy_assign_throws<small_throws_on_copy>();
    test_copy_assign_throws<large_throws_on_copy>();

  return 0;
}
