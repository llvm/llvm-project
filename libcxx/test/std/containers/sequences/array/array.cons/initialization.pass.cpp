//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test all the ways of initializing a std::array.

#include <array>
#include <cassert>
#include <type_traits>
#include "test_macros.h"


struct NoDefault {
    TEST_CONSTEXPR NoDefault(int) { }
};

// Test default initialization
// This one isn't constexpr because omitting to initialize fundamental types
// isn't valid in a constexpr context.
struct test_default_initialization {
    template <typename T>
    void operator()() const
    {
        std::array<T, 0> a0; (void)a0;
        std::array<T, 1> a1; (void)a1;
        std::array<T, 2> a2; (void)a2;
        std::array<T, 3> a3; (void)a3;

        std::array<NoDefault, 0> nodefault; (void)nodefault;
    }
};

struct test_nondefault_initialization {
    template <typename T>
    TEST_CONSTEXPR_CXX14 void operator()() const
    {
        // Check direct-list-initialization syntax (introduced in C++11)
    #if TEST_STD_VER >= 11
        {
            {
                std::array<T, 0> a0_0{}; (void)a0_0;
            }
            {
                std::array<T, 1> a1_0{}; (void)a1_0;
                std::array<T, 1> a1_1{T()}; (void)a1_1;
            }
            {
                std::array<T, 2> a2_0{}; (void)a2_0;
                std::array<T, 2> a2_1{T()}; (void)a2_1;
                std::array<T, 2> a2_2{T(), T()}; (void)a2_2;
            }
            {
                std::array<T, 3> a3_0{}; (void)a3_0;
                std::array<T, 3> a3_1{T()}; (void)a3_1;
                std::array<T, 3> a3_2{T(), T()}; (void)a3_2;
                std::array<T, 3> a3_3{T(), T(), T()}; (void)a3_3;
            }

            std::array<NoDefault, 0> nodefault{}; (void)nodefault;
        }
    #endif

        // Check copy-list-initialization syntax
        {
            {
                std::array<T, 0> a0_0 = {}; (void)a0_0;
            }
            {
                std::array<T, 1> a1_0 = {}; (void)a1_0;
                std::array<T, 1> a1_1 = {T()}; (void)a1_1;
            }
            {
                std::array<T, 2> a2_0 = {}; (void)a2_0;
                std::array<T, 2> a2_1 = {T()}; (void)a2_1;
                std::array<T, 2> a2_2 = {T(), T()}; (void)a2_2;
            }
            {
                std::array<T, 3> a3_0 = {}; (void)a3_0;
                std::array<T, 3> a3_1 = {T()}; (void)a3_1;
                std::array<T, 3> a3_2 = {T(), T()}; (void)a3_2;
                std::array<T, 3> a3_3 = {T(), T(), T()}; (void)a3_3;
            }

            std::array<NoDefault, 0> nodefault = {}; (void)nodefault;
        }

        // Test aggregate initialization
        {
            {
                std::array<T, 0> a0_0 = {{}}; (void)a0_0;
            }
            {
                std::array<T, 1> a1_0 = {{}}; (void)a1_0;
                std::array<T, 1> a1_1 = {{T()}}; (void)a1_1;
            }
            {
                std::array<T, 2> a2_0 = {{}}; (void)a2_0;
                std::array<T, 2> a2_1 = {{T()}}; (void)a2_1;
                std::array<T, 2> a2_2 = {{T(), T()}}; (void)a2_2;
            }
            {
                std::array<T, 3> a3_0 = {{}}; (void)a3_0;
                std::array<T, 3> a3_1 = {{T()}}; (void)a3_1;
                std::array<T, 3> a3_2 = {{T(), T()}}; (void)a3_2;
                std::array<T, 3> a3_3 = {{T(), T(), T()}}; (void)a3_3;
            }

            // See http://wg21.link/LWG2157
            std::array<NoDefault, 0> nodefault = {{}}; (void)nodefault;
        }
    }
};

// Test construction from an initializer-list
TEST_CONSTEXPR_CXX14 bool test_initializer_list()
{
    {
        std::array<double, 3> const a3_0 = {};
        assert(a3_0[0] == double());
        assert(a3_0[1] == double());
        assert(a3_0[2] == double());
    }
    {
        std::array<double, 3> const a3_1 = {1};
        assert(a3_1[0] == double(1));
        assert(a3_1[1] == double());
        assert(a3_1[2] == double());
    }
    {
        std::array<double, 3> const a3_2 = {1, 2.2};
        assert(a3_2[0] == double(1));
        assert(a3_2[1] == 2.2);
        assert(a3_2[2] == double());
    }
    {
        std::array<double, 3> const a3_3 = {1, 2, 3.5};
        assert(a3_3[0] == double(1));
        assert(a3_3[1] == double(2));
        assert(a3_3[2] == 3.5);
    }

    return true;
}

struct Empty { };
struct Trivial { int i; int j; };
struct NonTrivial {
    TEST_CONSTEXPR NonTrivial() { }
    TEST_CONSTEXPR NonTrivial(NonTrivial const&) { }
};
struct NonEmptyNonTrivial {
    int i; int j;
    TEST_CONSTEXPR NonEmptyNonTrivial() : i(22), j(33) { }
    TEST_CONSTEXPR NonEmptyNonTrivial(NonEmptyNonTrivial const&) : i(22), j(33) { }
};

template <typename F>
TEST_CONSTEXPR_CXX14 bool with_all_types()
{
    F().template operator()<char>();
    F().template operator()<int>();
    F().template operator()<long>();
    F().template operator()<float>();
    F().template operator()<double>();
    F().template operator()<long double>();
    F().template operator()<Empty>();
    F().template operator()<Trivial>();
    F().template operator()<NonTrivial>();
    F().template operator()<NonEmptyNonTrivial>();
    return true;
}

int main(int, char**)
{
    with_all_types<test_nondefault_initialization>();
    with_all_types<test_default_initialization>(); // not constexpr
    test_initializer_list();
#if TEST_STD_VER >= 14
    static_assert(with_all_types<test_nondefault_initialization>(), "");
    static_assert(test_initializer_list(), "");
#endif

    return 0;
}
