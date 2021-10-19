//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// multiplies

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::multiplies<int> F;
    const F f = F();
#if TEST_STD_VER <= 17
    static_assert((std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((std::is_same<int, F::result_type>::value), "" );
#endif
    assert(f(3, 2) == 6);
#if TEST_STD_VER > 11
    typedef std::multiplies<> F2;
    const F2 f2 = F2();
    assert(f2(3,2) == 6);
    assert(f2(3.0, 2) == 6);
    assert(f2(3, 2.5) == 7.5); // exact in binary

    constexpr int foo = std::multiplies<int> () (3, 2);
    static_assert ( foo == 6, "" );

    constexpr double bar = std::multiplies<> () (3.0, 2);
    static_assert ( bar == 6.0, "" );
#endif

  return 0;
}
