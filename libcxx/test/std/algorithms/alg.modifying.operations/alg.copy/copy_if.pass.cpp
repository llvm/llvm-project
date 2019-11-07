//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter,
//          Predicate<auto, InIter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr OutIter   // constexpr after C++17
//   copy_if(InIter first, InIter last, OutIter result, Pred pred);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

struct Pred
{
    TEST_CONSTEXPR_CXX14 bool operator()(int i) {return i % 3 == 0;}
};

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 void
test_copy_if()
{
    const unsigned N = 1000;
    int ia[N] = {};
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    int ib[N] = {0};

    OutIter r = std::copy_if(InIter(ia), InIter(ia+N), OutIter(ib), Pred());
    assert(base(r) == ib+N/3+1);
    for (unsigned i = 0; i < N/3+1; ++i)
        assert(ib[i] % 3 == 0);
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    test_copy_if<input_iterator<const int*>, output_iterator<int*> >();
    test_copy_if<input_iterator<const int*>, input_iterator<int*> >();
    test_copy_if<input_iterator<const int*>, forward_iterator<int*> >();
    test_copy_if<input_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_if<input_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_if<input_iterator<const int*>, int*>();

    test_copy_if<forward_iterator<const int*>, output_iterator<int*> >();
    test_copy_if<forward_iterator<const int*>, input_iterator<int*> >();
    test_copy_if<forward_iterator<const int*>, forward_iterator<int*> >();
    test_copy_if<forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_if<forward_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_if<forward_iterator<const int*>, int*>();

    test_copy_if<bidirectional_iterator<const int*>, output_iterator<int*> >();
    test_copy_if<bidirectional_iterator<const int*>, input_iterator<int*> >();
    test_copy_if<bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test_copy_if<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_if<bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_if<bidirectional_iterator<const int*>, int*>();

    test_copy_if<random_access_iterator<const int*>, output_iterator<int*> >();
    test_copy_if<random_access_iterator<const int*>, input_iterator<int*> >();
    test_copy_if<random_access_iterator<const int*>, forward_iterator<int*> >();
    test_copy_if<random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_if<random_access_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_if<random_access_iterator<const int*>, int*>();

    test_copy_if<const int*, output_iterator<int*> >();
    test_copy_if<const int*, input_iterator<int*> >();
    test_copy_if<const int*, forward_iterator<int*> >();
    test_copy_if<const int*, bidirectional_iterator<int*> >();
    test_copy_if<const int*, random_access_iterator<int*> >();
    test_copy_if<const int*, int*>();

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
