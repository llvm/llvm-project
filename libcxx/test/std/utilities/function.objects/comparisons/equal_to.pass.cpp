//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// equal_to

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::equal_to<int> F;
    const F f = F();
    static_assert((std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((std::is_same<bool, F::result_type>::value), "" );
    assert(f(36, 36));
    assert(!f(36, 6));
#if TEST_STD_VER > 11
    typedef std::equal_to<> F2;
    const F2 f2 = F2();
    assert(f2(36, 36));
    assert(!f2(36, 6));
    assert(f2(36, 36.0));
    assert(f2(36.0, 36L));

    constexpr bool foo = std::equal_to<int> () (36, 36);
    static_assert ( foo, "" );

    constexpr bool bar = std::equal_to<> () (36.0, 36);
    static_assert ( bar, "" );
#endif

  return 0;
}
