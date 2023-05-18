//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:

// template <class T1, class T2>
//   constexpr bool
//   operator==(const allocator<T1>&, const allocator<T2>&) throw();
//
// template <class T1, class T2>
//   constexpr bool
//   operator!=(const allocator<T1>&, const allocator<T2>&) throw();

#include <memory>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test()
{
    std::allocator<int> a1;
    std::allocator<int> a2;
    assert(a1 == a2);
    assert(!(a1 != a2));

    return true;
}

int main(int, char**)
{
    test();

#if TEST_STD_VER > 17
    static_assert(test());
#endif

    return 0;
}
