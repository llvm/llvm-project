//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <iterator>

// move_iterator

// pointer operator->() const;
//
//  constexpr in C++17
//  deprecated in C++20

#include <iterator>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX17 bool test()
{
    char a[] = "123456789";
    std::move_iterator<char *> it1 = std::make_move_iterator(a);
    std::move_iterator<char *> it2 = std::make_move_iterator(a + 1);
    assert(it1.operator->() == a);
    assert(it2.operator->() == a + 1);

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 14
    static_assert(test());
#endif

    return 0;
}
