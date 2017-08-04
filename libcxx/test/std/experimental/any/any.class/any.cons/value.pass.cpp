//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// XFAIL: availability=macosx

// <experimental/any>

// template <class Value> any(Value &&)

// Test construction from a value.
// Concerns:
// ---------
// 1. The value is properly move/copied depending on the value category.
// 2. Both small and large values are properly handled.


#include <experimental/any>
#include <cassert>

#include "experimental_any_helpers.h"
#include "count_new.hpp"
#include "test_macros.h"

using std::experimental::any;
using std::experimental::any_cast;

template <class Type>
void test_copy_value_throws()
{
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    assert(Type::count == 0);
    {
        Type const t(42);
        assert(Type::count == 1);
        try {
            any const a2(t);
            assert(false);
        } catch (my_any_exception const &) {
            // do nothing
        } catch (...) {
            assert(false);
        }
        assert(Type::count == 1);
        assert(t.value == 42);
    }
    assert(Type::count == 0);
#endif
}

void test_move_value_throws()
{
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    assert(throws_on_move::count == 0);
    {
        throws_on_move v;
        assert(throws_on_move::count == 1);
        try {
            any const a(std::move(v));
            assert(false);
        } catch (my_any_exception const &) {
            // do nothing
        } catch (...) {
            assert(false);
        }
        assert(throws_on_move::count == 1);
    }
    assert(throws_on_move::count == 0);
#endif
}

template <class Type>
void test_copy_move_value() {
    // constructing from a small type should perform no allocations.
    DisableAllocationGuard g(isSmallType<Type>()); ((void)g);
    assert(Type::count == 0);
    Type::reset();
    {
        Type t(42);
        assert(Type::count == 1);

        any a(t);

        assert(Type::count == 2);
        assert(Type::copied == 1);
        assert(Type::moved == 0);
        assertContains<Type>(a, 42);
    }
    assert(Type::count == 0);
    Type::reset();
    {
        Type t(42);
        assert(Type::count == 1);

        any a(std::move(t));

        assert(Type::count == 2);
        assert(Type::copied == 0);
        assert(Type::moved == 1);
        assertContains<Type>(a, 42);
    }
}


int main() {
    test_copy_move_value<small>();
    test_copy_move_value<large>();
    test_copy_value_throws<small_throws_on_copy>();
    test_copy_value_throws<large_throws_on_copy>();
    test_move_value_throws();
}
