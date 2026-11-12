//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<BidirectionalIterator Iter>
//   requires HasSwap<Iter::reference, Iter::reference>
//   constexpr void  // constexpr in C++20
//   reverse(Iter first, Iter last);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter>
TEST_CONSTEXPR_CXX20 bool
test()
{
    int ia[] = {0};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    std::reverse(Iter(ia), Iter(ia));
    assert(ia[0] == 0);
    std::reverse(Iter(ia), Iter(ia+sa));
    assert(ia[0] == 0);

    int ib[] = {0, 1};
    const unsigned sb = sizeof(ib)/sizeof(ib[0]);
    std::reverse(Iter(ib), Iter(ib+sb));
    assert(ib[0] == 1);
    assert(ib[1] == 0);

    int ic[] = {0, 1, 2};
    const unsigned sc = sizeof(ic)/sizeof(ic[0]);
    std::reverse(Iter(ic), Iter(ic+sc));
    assert(ic[0] == 2);
    assert(ic[1] == 1);
    assert(ic[2] == 0);

    int id[] = {0, 1, 2, 3};
    const unsigned sd = sizeof(id)/sizeof(id[0]);
    std::reverse(Iter(id), Iter(id+sd));
    assert(id[0] == 3);
    assert(id[1] == 2);
    assert(id[2] == 1);
    assert(id[3] == 0);

    return true;
}

int main(int, char**)
{
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();

#if TEST_STD_VER >= 20
    static_assert(test<bidirectional_iterator<int*>>());
    static_assert(test<random_access_iterator<int*>>());
    static_assert(test<int*>());
#endif

    return 0;
}
