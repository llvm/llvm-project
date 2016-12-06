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

// Test the return type deduction.

#include <experimental/tuple>
#include <cassert>

static int my_int = 42;

template <int N> struct index_t {};

void f(index_t<0>) {}

int f(index_t<1>) { return 0; }

int & f(index_t<2>) { return static_cast<int &>(my_int); }
int const & f(index_t<3>) { return static_cast<int const &>(my_int); }
int volatile & f(index_t<4>) { return static_cast<int volatile &>(my_int); }
int const volatile & f(index_t<5>) { return static_cast<int const volatile &>(my_int); }

int && f(index_t<6>) { return static_cast<int &&>(my_int); }
int const && f(index_t<7>) { return static_cast<int const &&>(my_int); }
int volatile && f(index_t<8>) { return static_cast<int volatile &&>(my_int); }
int const volatile && f(index_t<9>) { return static_cast<int const volatile &&>(my_int); }

int * f(index_t<10>) { return static_cast<int *>(&my_int); }
int const * f(index_t<11>) { return static_cast<int const *>(&my_int); }
int volatile * f(index_t<12>) { return static_cast<int volatile *>(&my_int); }
int const volatile * f(index_t<13>) { return static_cast<int const volatile *>(&my_int); }


template <int Func, class Expect>
void test()
{
    using F = decltype(f(index_t<Func>{}));
    static_assert(std::is_same<F, Expect>::value, "");
}

namespace ex = std::experimental;

int main()
{
    test<0, void>();
    test<1, int>();
    test<2, int &>();
    test<3, int const &>();
    test<4, int volatile &>();
    test<5, int const volatile &>();
    test<6, int &&>();
    test<7, int const &&>();
    test<8, int volatile &&>();
    test<9, int const volatile &&>();
    test<10, int *>();
    test<11, int const *>();
    test<12, int volatile *>();
    test<13, int const volatile *>();
}
