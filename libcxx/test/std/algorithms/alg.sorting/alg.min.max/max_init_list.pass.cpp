//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// template <class T>
//   T
//   max(initializer_list<T> t);

#include <algorithm>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    int i = std::max({2, 3, 1});
    assert(i == 3);
    i = std::max({2, 1, 3});
    assert(i == 3);
    i = std::max({3, 1, 2});
    assert(i == 3);
    i = std::max({3, 2, 1});
    assert(i == 3);
    i = std::max({1, 2, 3});
    assert(i == 3);
    i = std::max({1, 3, 2});
    assert(i == 3);
#if TEST_STD_VER >= 14
    {
    static_assert(std::max({1, 3, 2}) == 3, "");
    static_assert(std::max({2, 1, 3}) == 3, "");
    static_assert(std::max({3, 2, 1}) == 3, "");
    }
#endif

  return 0;
}
