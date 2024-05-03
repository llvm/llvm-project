//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// any& operator=(any &&);

// Test move assignment.

#include <any>
#include <cassert>

#include "any_helpers.h"
#include "test_macros.h"

template <class LHS, class RHS>
void test_move_assign() {
    assert(LHS::count == 0);
    assert(RHS::count == 0);
    {
        LHS const s1(1);
        std::any a = s1;
        RHS const s2(2);
        std::any a2 = s2;

        assert(LHS::count == 2);
        assert(RHS::count == 2);

        a = std::move(a2);

        assert(LHS::count == 1);
        assert(RHS::count == 2 + a2.has_value());
        LIBCPP_ASSERT(RHS::count == 2); // libc++ leaves the object empty

        assertContains<RHS>(a, 2);
        if (a2.has_value())
            assertContains<RHS>(a2, 0);
        LIBCPP_ASSERT(!a2.has_value());
    }
    assert(LHS::count == 0);
    assert(RHS::count == 0);
}

template <class LHS>
void test_move_assign_empty() {
    assert(LHS::count == 0);
    {
        std::any a;
        std::any a2 = LHS(1);

        assert(LHS::count == 1);

        a = std::move(a2);

        assert(LHS::count == 1 + a2.has_value());
        LIBCPP_ASSERT(LHS::count == 1);

        assertContains<LHS>(a, 1);
        if (a2.has_value())
            assertContains<LHS>(a2, 0);
        LIBCPP_ASSERT(!a2.has_value());
    }
    assert(LHS::count == 0);
    {
        std::any a = LHS(1);
        std::any a2;

        assert(LHS::count == 1);

        a = std::move(a2);

        assert(LHS::count == 0);

        assertEmpty<LHS>(a);
        assertEmpty(a2);
    }
    assert(LHS::count == 0);
}

void test_move_assign_noexcept() {
    std::any a1;
    std::any a2;
    ASSERT_NOEXCEPT(a1 = std::move(a2));
}

int main(int, char**) {
    test_move_assign_noexcept();
    test_move_assign<small1, small2>();
    test_move_assign<large1, large2>();
    test_move_assign<small, large>();
    test_move_assign<large, small>();
    test_move_assign_empty<small>();
    test_move_assign_empty<large>();

  return 0;
}
