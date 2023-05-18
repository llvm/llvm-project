//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <BackInsertionContainer Cont>
//   back_insert_iterator<Cont>
//   back_inserter(Cont& x); // constexpr in C++20

// template <BackInsertionContainer Cont>
//   front_insert_iterator<Cont>
//   front_inserter(Cont& x); // constexpr in C++20

#include <cassert>
#include <iterator>
#include <list>

#include "test_macros.h"
#include "nasty_containers.h"
#include "test_constexpr_container.h"

template <class C>
TEST_CONSTEXPR_CXX20 bool
test(C c)
{
    std::back_insert_iterator<C> i = std::back_inserter(c);
    i = 3;
    assert(c.size() == 1);
    assert(c.back() == 3);
    i = 4;
    assert(c.size() == 2);
    assert(c.back() == 4);
    return true;
}

int main(int, char**)
{
    test(std::vector<int>());
    test(nasty_vector<int>());
#if TEST_STD_VER >= 20
    test(ConstexprFixedCapacityDeque<int, 10>());
    static_assert(test(ConstexprFixedCapacityDeque<int, 10>()));
#endif
    return 0;
}
