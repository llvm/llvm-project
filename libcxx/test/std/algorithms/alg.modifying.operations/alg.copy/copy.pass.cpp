//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter   // constexpr after C++17
//   copy(InIter first, InIter last, OutIter result);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 void
test_copy()
{
    const unsigned N = 1000;
    int ia[N] = {};
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    int ib[N] = {0};

    OutIter r = std::copy(InIter(ia), InIter(ia+N), OutIter(ib));
    assert(base(r) == ib+N);
    for (unsigned i = 0; i < N; ++i)
        assert(ia[i] == ib[i]);
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    test_copy<cpp17_input_iterator<const int*>, cpp17_output_iterator<int*> >();
    test_copy<cpp17_input_iterator<const int*>, forward_iterator<int*> >();
    test_copy<cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy<cpp17_input_iterator<const int*>, random_access_iterator<int*> >();
    test_copy<cpp17_input_iterator<const int*>, int*>();

    test_copy<forward_iterator<const int*>, cpp17_output_iterator<int*> >();
    test_copy<forward_iterator<const int*>, forward_iterator<int*> >();
    test_copy<forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy<forward_iterator<const int*>, random_access_iterator<int*> >();
    test_copy<forward_iterator<const int*>, int*>();

    test_copy<bidirectional_iterator<const int*>, cpp17_output_iterator<int*> >();
    test_copy<bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test_copy<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy<bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test_copy<bidirectional_iterator<const int*>, int*>();

    test_copy<random_access_iterator<const int*>, cpp17_output_iterator<int*> >();
    test_copy<random_access_iterator<const int*>, forward_iterator<int*> >();
    test_copy<random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy<random_access_iterator<const int*>, random_access_iterator<int*> >();
    test_copy<random_access_iterator<const int*>, int*>();

    test_copy<const int*, cpp17_output_iterator<int*> >();
    test_copy<const int*, forward_iterator<int*> >();
    test_copy<const int*, bidirectional_iterator<int*> >();
    test_copy<const int*, random_access_iterator<int*> >();
    test_copy<const int*, int*>();

#if TEST_STD_VER > 17
    test_copy<cpp17_input_iterator<const int*>, contiguous_iterator<int*>>();
    test_copy<forward_iterator<const int*>, contiguous_iterator<int*>>();
    test_copy<bidirectional_iterator<const int*>, contiguous_iterator<int*>>();
    test_copy<random_access_iterator<const int*>, contiguous_iterator<int*>>();
    test_copy<const int*, contiguous_iterator<int*>>();

    test_copy<contiguous_iterator<const int*>, cpp17_output_iterator<int*>>();
    test_copy<contiguous_iterator<const int*>, forward_iterator<int*>>();
    test_copy<contiguous_iterator<const int*>, bidirectional_iterator<int*>>();
    test_copy<contiguous_iterator<const int*>, random_access_iterator<int*>>();
    test_copy<contiguous_iterator<const int*>, int*>();
#endif

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
