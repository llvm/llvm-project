//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <utility>

// template <class T1, class T2> struct pair

// struct piecewise_construct_t { explicit piecewise_construct_t() = default; };
// constexpr piecewise_construct_t piecewise_construct = piecewise_construct_t();

#include <utility>
#include <tuple>
#include <cassert>

#include "test_macros.h"

class A
{
    int i_;
    char c_;
public:
    constexpr A(int i, char c) : i_(i), c_(c) {}
    constexpr int get_i() const {return i_;}
    constexpr char get_c() const {return c_;}
};

class B
{
    double d_;
    unsigned u1_;
    unsigned u2_;
public:
    constexpr explicit B(double d, unsigned u1, unsigned u2) : d_(d), u1_(u1), u2_(u2) {}
    constexpr double get_d() const {return d_;}
    constexpr unsigned get_u1() const {return u1_;}
    constexpr unsigned get_u2() const {return u2_;}
};

TEST_CONSTEXPR_CXX20
bool test()
{
    std::pair<A, B> p(std::piecewise_construct,
                      std::make_tuple(4, 'a'),
                      std::make_tuple(3.5, 6u, 2u));
    assert(p.first.get_i() == 4);
    assert(p.first.get_c() == 'a');
    assert(p.second.get_d() == 3.5);
    assert(p.second.get_u1() == 6u);
    assert(p.second.get_u2() == 2u);

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER >= 20
    static_assert(test());
#endif

    return 0;
}
