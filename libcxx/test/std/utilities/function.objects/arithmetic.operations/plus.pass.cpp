//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// plus

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::plus<int> F;
    const F f = F();
#if TEST_STD_VER <= 17
    static_assert((std::is_same<int, F::first_argument_type>::value), "");
    static_assert((std::is_same<int, F::second_argument_type>::value), "");
    static_assert((std::is_same<int, F::result_type>::value), "");
#endif
    assert(f(3, 2) == 5);
#if TEST_STD_VER > 11
    typedef std::plus<> F2;
    const F2 f2 = F2();
    assert(f2(3,2) == 5);
    assert(f2(3.0, 2) == 5);
    assert(f2(3, 2.5) == 5.5);

    constexpr int foo = std::plus<int> () (3, 2);
    static_assert ( foo == 5, "" );

    constexpr double bar = std::plus<> () (3.0, 2);
    static_assert ( bar == 5.0, "" );
#endif

  return 0;
}
