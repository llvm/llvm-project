//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <InsertionContainer Cont>
//   insert_iterator<Cont>
//   inserter(Cont& x, Cont::iterator i);

#include <cassert>
#include <iterator>
#include <vector>

#include "test_macros.h"
#include "nasty_containers.h"
#include "test_constexpr_container.h"

template <class C>
TEST_CONSTEXPR_CXX20 bool
test(C c)
{
    std::insert_iterator<C> i = std::inserter(c, c.end());
    i = 0;
    assert(c.size() == 1);
    assert(c.back() == 0);
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
