//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// pointer operator->() const;
//
//  constexpr in C++17
//  removed in C++20

#include <iterator>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER > 17
template <class T>
concept HasArrow = requires (T t) {
    t.operator->();
};
static_assert(!HasArrow<std::move_iterator<int*>>);
static_assert(!HasArrow<std::move_iterator<int*>&>);
static_assert(!HasArrow<std::move_iterator<int*>&&>);
#endif // TEST_STD_VER > 17

TEST_CONSTEXPR_CXX17 bool test()
{
#if TEST_STD_VER <= 17
    char a[] = "123456789";
    std::move_iterator<char *> it1 = std::make_move_iterator(a);
    std::move_iterator<char *> it2 = std::make_move_iterator(a + 1);
    assert(it1.operator->() == a);
    assert(it2.operator->() == a + 1);
#endif
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
