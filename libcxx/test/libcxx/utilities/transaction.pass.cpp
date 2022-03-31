//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <utility> // for __transaction
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test() {
    // Make sure the transaction is rolled back if it is not marked as complete when
    // it goes out of scope.
    {
        bool rolled_back = false;
        {
            auto rollback = [&] { rolled_back = true; };
            std::__transaction<decltype(rollback)> t(rollback);
        }
        assert(rolled_back);
    }

    // Make sure the transaction is not rolled back if it is marked as complete when
    // it goes out of scope.
    {
        bool rolled_back = false;
        {
            auto rollback = [&] { rolled_back = true; };
            std::__transaction<decltype(rollback)> t(rollback);
            t.__complete();
        }
        assert(!rolled_back);
    }

    // Make sure that we will perform the right number of rollbacks when a transaction has
    // been moved around
    {
        // When we don't complete it (exactly 1 rollback should happen)
        {
            int rollbacks = 0;
            {
                auto rollback = [&] { ++rollbacks; };
                std::__transaction<decltype(rollback)> t(rollback);
                auto other = std::move(t);
            }
            assert(rollbacks == 1);
        }

        // When we do complete it (no rollbacks should happen)
        {
            int rollbacks = 0;
            {
                auto rollback = [&] { ++rollbacks; };
                std::__transaction<decltype(rollback)> t(rollback);
                auto other = std::move(t);
                other.__complete();
            }
            assert(rollbacks == 0);
        }
    }

    // Basic properties of the type
    {
        struct Rollback { void operator()() const { } };
        using Transaction = std::__transaction<Rollback>;

        static_assert(!std::is_default_constructible<Transaction>::value, "");

        static_assert(!std::is_copy_constructible<Transaction>::value, "");
        static_assert( std::is_move_constructible<Transaction>::value, "");

        static_assert(!std::is_copy_assignable<Transaction>::value, "");
        static_assert(!std::is_move_assignable<Transaction>::value, "");

        // Check noexcept-ness of a few operations
        {
            struct ThrowOnMove {
                ThrowOnMove(ThrowOnMove&&) noexcept(false) { }
                void operator()() const { }
            };
            using ThrowOnMoveTransaction = std::__transaction<ThrowOnMove>;

            ASSERT_NOEXCEPT(std::declval<Transaction>().__complete());
            static_assert( std::is_nothrow_move_constructible<Transaction>::value, "");
            static_assert(!std::is_nothrow_move_constructible<ThrowOnMoveTransaction>::value, "");
        }
    }

    return true;
}

void test_exceptions() {
#ifndef TEST_HAS_NO_EXCEPTIONS
    // Make sure the rollback is performed when an exception is thrown during the
    // lifetime of the transaction.
    {
        bool rolled_back = false;
        auto rollback = [&] { rolled_back = true; };
        try {
            std::__transaction<decltype(rollback)> t(rollback);
            throw 0;
        } catch (...) { }
        assert(rolled_back);
    }

    // Make sure we don't roll back if an exception is thrown but the transaction
    // has been marked as complete when that happens.
    {
        bool rolled_back = false;
        auto rollback = [&] { rolled_back = true; };
        try {
            std::__transaction<decltype(rollback)> t(rollback);
            t.__complete();
            throw 0;
        } catch (...) { }
        assert(!rolled_back);
    }

    // Make sure __transaction does not rollback if the transaction is marked as
    // completed within a destructor.
    {
        struct S {
            explicit S(bool& x) : x_(x) { }

            ~S() {
                auto rollback = [this]{ x_ = true; };
                std::__transaction<decltype(rollback)> t(rollback);
                t.__complete();
            }

            bool& x_;
        };

        bool rolled_back = false;
        try {
            S s(rolled_back);
            throw 0;
        } catch (...) {
            assert(!rolled_back);
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
    test();
    test_exceptions();
#if TEST_STD_VER > 17
    static_assert(test(), "");
#endif
    return 0;
}
