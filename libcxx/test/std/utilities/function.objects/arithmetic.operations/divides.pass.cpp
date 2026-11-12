//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// divides

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::divides<int> F;
    const F f = F();
#if TEST_STD_VER <= 17
    static_assert((std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((std::is_same<int, F::result_type>::value), "" );
#endif
    assert(f(36, 4) == 9);
#if TEST_STD_VER > 11
    typedef std::divides<> F2;
    const F2 f2 = F2();
    assert(f2(36, 4) == 9);
    assert(f2(36.0, 4) == 9);
    assert(f2(18, 4.0) == 4.5); // exact in binary

    constexpr int foo = std::divides<int> () (3, 2);
    static_assert ( foo == 1, "" );

    constexpr double bar = std::divides<> () (3.0, 2);
    static_assert ( bar == 1.5, "" ); // exact in binary
#endif

  return 0;
}
