//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// constexpr common_type_t<duration> operator-() const;

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    const std::chrono::minutes m(3);
    std::chrono::minutes m2 = -m;
    assert(m2.count() == -m.count());
    }
#if TEST_STD_VER >= 11
    {
    constexpr std::chrono::minutes m(3);
    constexpr std::chrono::minutes m2 = -m;
    static_assert(m2.count() == -m.count(), "");
    }
#endif

// P0548
    {
    typedef std::chrono::duration<int, std::ratio<10,10> > D10;
    typedef std::chrono::duration<int, std::ratio< 1, 1> > D1;
    D10 zero(0);
    D10 one(1);
    static_assert( (std::is_same< decltype(-one), decltype(zero-one) >::value), "");
    static_assert( (std::is_same< decltype(zero-one), D1>::value), "");
    static_assert( (std::is_same< decltype(-one),     D1>::value), "");
    static_assert( (std::is_same< decltype(+one),     D1>::value), "");
    }

  return 0;
}
