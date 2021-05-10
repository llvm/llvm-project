//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_any_cast is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.12 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9 && !no-exceptions

// <any>

// any(any &&) noexcept;

#include <any>
#include <utility>
#include <type_traits>
#include <cassert>

#include "any_helpers.h"
#include "count_new.h"
#include "test_macros.h"

using std::any;
using std::any_cast;

// Moves are always noexcept. The throws_on_move object
// must be stored dynamically so the pointer is moved and
// not the stored object.
void test_move_does_not_throw()
{
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    assert(throws_on_move::count == 0);
    {
        throws_on_move v(42);
        any a(v);
        assert(throws_on_move::count == 2);
        // No allocations should be performed after this point.
        DisableAllocationGuard g; ((void)g);
        try {
            any const a2(std::move(a));
            assertEmpty(a);
            assertContains<throws_on_move>(a2, 42);
        } catch (...) {
            assert(false);
        }
        assert(throws_on_move::count == 1);
        assertEmpty(a);
    }
    assert(throws_on_move::count == 0);
#endif
}

void test_move_empty() {
    DisableAllocationGuard g; ((void)g); // no allocations should be performed.

    any a1;
    any a2(std::move(a1));

    assertEmpty(a1);
    assertEmpty(a2);
}

template <class Type>
void test_move() {
    assert(Type::count == 0);
    Type::reset();
    {
        any a((Type(42)));
        assert(Type::count == 1);
        assert(Type::copied == 0);
        assert(Type::moved == 1);

        // Moving should not perform allocations since it must be noexcept.
        DisableAllocationGuard g; ((void)g);

        any a2(std::move(a));

        assert(Type::moved == 1 || Type::moved == 2); // zero or more move operations can be performed.
        assert(Type::copied == 0); // no copies can be performed.
        assert(Type::count == 1 + a.has_value());
        assertContains<Type>(a2, 42);
        LIBCPP_ASSERT(!a.has_value()); // Moves are always destructive.
        if (a.has_value())
            assertContains<Type>(a, 0);
    }
    assert(Type::count == 0);
}

int main(int, char**)
{
    // noexcept test
    {
        static_assert(
            std::is_nothrow_move_constructible<any>::value
          , "any must be nothrow move constructible"
          );
    }
    test_move<small>();
    test_move<large>();
    test_move_empty();
    test_move_does_not_throw();

  return 0;
}
