//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2, typename Compare>
//   requires Predicate<Compare, Iter1::value_type, Iter2::value_type>
//         && Predicate<Compare, Iter2::value_type, Iter1::value_type>
//   constexpr bool             // constexpr after C++17
//   includes(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Compare comp);

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter1, class Iter2>
TEST_CONSTEXPR_CXX20
void test()
{
    int ia[] = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    int ib[] = {4, 2};
    const unsigned sb = sizeof(ib)/sizeof(ib[0]);
    int ic[] = {2, 1};
    const unsigned sc = sizeof(ic)/sizeof(ic[0]); ((void)sc);
    int id[] = {3, 3, 3, 3};
    const unsigned sd = sizeof(id)/sizeof(id[0]); ((void)sd);

    assert(std::includes(Iter1(ia), Iter1(ia), Iter2(ib), Iter2(ib), std::greater<int>()));
    assert(!std::includes(Iter1(ia), Iter1(ia), Iter2(ib), Iter2(ib+1), std::greater<int>()));
    assert(std::includes(Iter1(ia), Iter1(ia+1), Iter2(ib), Iter2(ib), std::greater<int>()));
    assert(std::includes(Iter1(ia), Iter1(ia+sa), Iter2(ia), Iter2(ia+sa), std::greater<int>()));

    assert(std::includes(Iter1(ia), Iter1(ia+sa), Iter2(ib), Iter2(ib+sb), std::greater<int>()));
    assert(!std::includes(Iter1(ib), Iter1(ib+sb), Iter2(ia), Iter2(ia+sa), std::greater<int>()));

    assert(std::includes(Iter1(ia+8), Iter1(ia+sa), Iter2(ic), Iter2(ic+sc), std::greater<int>()));
    assert(!std::includes(Iter1(ia+8), Iter1(ia+sa), Iter2(ib), Iter2(ib+sb), std::greater<int>()));

    assert(std::includes(Iter1(ia), Iter1(ia+sa), Iter2(id), Iter2(id+1), std::greater<int>()));
    assert(std::includes(Iter1(ia), Iter1(ia+sa), Iter2(id), Iter2(id+2), std::greater<int>()));
    assert(std::includes(Iter1(ia), Iter1(ia+sa), Iter2(id), Iter2(id+3), std::greater<int>()));
    assert(!std::includes(Iter1(ia), Iter1(ia+sa), Iter2(id), Iter2(id+4), std::greater<int>()));
}

TEST_CONSTEXPR_CXX20
bool do_tests()
{
    test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, const int*>();

    test<forward_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<forward_iterator<const int*>, forward_iterator<const int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<forward_iterator<const int*>, random_access_iterator<const int*> >();
    test<forward_iterator<const int*>, const int*>();

    test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, const int*>();

    test<random_access_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<random_access_iterator<const int*>, forward_iterator<const int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<const int*> >();
    test<random_access_iterator<const int*>, const int*>();

    test<const int*, cpp17_input_iterator<const int*> >();
    test<const int*, forward_iterator<const int*> >();
    test<const int*, bidirectional_iterator<const int*> >();
    test<const int*, random_access_iterator<const int*> >();
    test<const int*, const int*>();

    return true;
}

int main(int, char**)
{
    do_tests();
#if TEST_STD_VER > 17
    static_assert(do_tests());
#endif
    return 0;
}
