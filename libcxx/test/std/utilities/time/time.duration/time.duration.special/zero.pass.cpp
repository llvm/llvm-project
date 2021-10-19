//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// static constexpr duration zero(); // noexcept after C++17

#include <chrono>
#include <cassert>

#include "test_macros.h"
#include "../../rep.h"

template <class D>
void test()
{
    LIBCPP_ASSERT_NOEXCEPT(std::chrono::duration_values<typename D::rep>::zero());
#if TEST_STD_VER > 17
    ASSERT_NOEXCEPT(       std::chrono::duration_values<typename D::rep>::zero());
#endif
    {
    typedef typename D::rep DRep;
    DRep zero_rep = std::chrono::duration_values<DRep>::zero();
    assert(D::zero().count() == zero_rep);
    }
#if TEST_STD_VER >= 11
    {
    typedef typename D::rep DRep;
    constexpr DRep zero_rep = std::chrono::duration_values<DRep>::zero();
    static_assert(D::zero().count() == zero_rep, "");
    }
#endif
}

int main(int, char**)
{
    test<std::chrono::duration<int> >();
    test<std::chrono::duration<Rep> >();

  return 0;
}
