//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/tuple>

// template <class F, class T> constexpr decltype(auto) apply(F &&, T &&)

// Testing constexpr evaluation

#include <experimental/tuple>
#include <utility>
#include <cassert>

constexpr int f_int_0() { return 1; }
constexpr int f_int_1(int x) { return  x; }
constexpr int f_int_2(int x, int y) { return (x + y); }

struct A_int_0
{
    constexpr A_int_0() {}
    constexpr int operator()() const { return 1; }
};

struct A_int_1
{
    constexpr A_int_1() {}
    constexpr int operator()(int x) const { return x; }
};

struct A_int_2
{
    constexpr A_int_2() {}
    constexpr int operator()(int x, int y) const { return (x + y); }
};

namespace ex = std::experimental;

template <class Tuple>
void test_0()
{
    // function
    {
        constexpr Tuple t{};
        static_assert(1 == ex::apply(f_int_0, t), "");
    }
    // function pointer
    {
        constexpr Tuple t{};
        constexpr auto fp = &f_int_0;
        static_assert(1 == ex::apply(fp, t), "");
    }
    // functor
    {
        constexpr Tuple t{};
        constexpr A_int_0 a;
        static_assert(1 == ex::apply(a, t), "");
    }
}

template <class Tuple>
void test_1()
{
    // function
    {
        constexpr Tuple t{1};
        static_assert(1 == ex::apply(f_int_1, t), "");
    }
    // function pointer
    {
        constexpr Tuple t{2};
        constexpr int (*fp)(int) = f_int_1;
        static_assert(2 == ex::apply(fp, t), "");
    }
    // functor
    {
        constexpr Tuple t{3};
        constexpr A_int_1 fn;
        static_assert(3 == ex::apply(fn, t), "");
    }
}

template <class Tuple>
void test_2()
{
    // function
    {
        constexpr Tuple t{1, 2};
        static_assert(3 == ex::apply(f_int_2, t), "");
    }
        // function pointer
    {
        constexpr Tuple t{2, 3};
        constexpr auto fp = &f_int_2;
        static_assert(5 == ex::apply(fp, t), "");
    }
    // functor
    {
        constexpr Tuple t{3, 4};
        constexpr A_int_2 a;
        static_assert(7 == ex::apply(a, t), "");
    }
}

int main()
{
    test_0<std::tuple<>>();
    test_1<std::tuple<int>>();
    test_2<std::tuple<int, int>>();
    test_2<std::pair<int, int>>();
}
