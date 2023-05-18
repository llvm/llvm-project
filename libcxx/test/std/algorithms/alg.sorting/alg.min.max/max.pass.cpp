//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<LessThanComparable T>
//   const T&
//   max(const T& a, const T& b);

#include <algorithm>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test(const T& a, const T& b, const T& x)
{
    assert(&std::max(a, b) == &x);
}

int main(int, char**)
{
    {
    int x = 0;
    int y = 0;
    test(x, y, x);
    test(y, x, y);
    }
    {
    int x = 0;
    int y = 1;
    test(x, y, y);
    test(y, x, y);
    }
    {
    int x = 1;
    int y = 0;
    test(x, y, x);
    test(y, x, x);
    }
#if TEST_STD_VER >= 14
    {
    constexpr int x = 1;
    constexpr int y = 0;
    static_assert(std::max(x, y) == x, "" );
    static_assert(std::max(y, x) == x, "" );
    }
#endif

  return 0;
}
