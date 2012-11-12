//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// static constexpr duration min();

#include <chrono>
#include <limits>
#include <cassert>

#include "../../rep.h"

template <class D>
void test()
{
    {
    typedef typename D::rep Rep;
    Rep min_rep = std::chrono::duration_values<Rep>::min();
    assert(D::min().count() == min_rep);
    }
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    {
    typedef typename D::rep Rep;
    constexpr Rep min_rep = std::chrono::duration_values<Rep>::min();
    static_assert(D::min().count() == min_rep, "");
    }
#endif
}

int main()
{
    test<std::chrono::duration<int> >();
    test<std::chrono::duration<Rep> >();
}
