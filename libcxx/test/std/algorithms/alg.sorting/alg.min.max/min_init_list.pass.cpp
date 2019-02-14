//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <algorithm>

// template<class T>
//   T
//   min(initializer_list<T> t);

#include <algorithm>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    int i = std::min({2, 3, 1});
    assert(i == 1);
    i = std::min({2, 1, 3});
    assert(i == 1);
    i = std::min({3, 1, 2});
    assert(i == 1);
    i = std::min({3, 2, 1});
    assert(i == 1);
    i = std::min({1, 2, 3});
    assert(i == 1);
    i = std::min({1, 3, 2});
    assert(i == 1);
#if TEST_STD_VER >= 14
    {
    static_assert(std::min({1, 3, 2}) == 1, "");
    static_assert(std::min({2, 1, 3}) == 1, "");
    static_assert(std::min({3, 2, 1}) == 1, "");
    }
#endif

  return 0;
}
