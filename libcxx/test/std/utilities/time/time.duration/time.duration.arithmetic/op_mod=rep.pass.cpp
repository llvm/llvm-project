//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// duration& operator%=(const rep& rhs)

#include <chrono>
#include <cassert>

#include "test_macros.h"

class NotARep {};

typedef std::chrono::seconds Duration;
Duration operator%=(Duration d, NotARep) { return d; }

#if TEST_STD_VER > 14
constexpr bool test_constexpr()
{
    std::chrono::seconds s(11);
    s %= 3;
    return s.count() == 2;
}
#endif

int main(int, char**)
{
    {
    std::chrono::microseconds us(11);
    us %= 3;
    assert(us.count() == 2);
    }

#if TEST_STD_VER > 14
    static_assert(test_constexpr(), "");
#endif

#if TEST_STD_VER >= 11
    { // This is PR#41130
    Duration d(5);
    NotARep n;
    d %= n;
    assert(d.count() == 5);
    }
#endif


  return 0;
}
