//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: clang-8

// <algorithm>

// template<BidirectionalIterator InIter, BidirectionalIterator OutIter>
//   requires OutputIterator<OutIter, InIter::reference>
//   constexpr OutIter   // constexpr after C++17
//   copy_backward(InIter first, InIter last, OutIter result);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "user_defined_integral.h"

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 void
test_copy_backward()
{
    const unsigned N = 1000;
    int ia[N] = {};
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    int ib[N] = {0};

    OutIter r = std::copy_backward(InIter(ia), InIter(ia+N), OutIter(ib+N));
    assert(base(r) == ib);
    for (unsigned i = 0; i < N; ++i)
        assert(ia[i] == ib[i]);
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    test_copy_backward<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_backward<bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_backward<bidirectional_iterator<const int*>, int*>();

    test_copy_backward<random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_backward<random_access_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_backward<random_access_iterator<const int*>, int*>();

    test_copy_backward<const int*, bidirectional_iterator<int*> >();
    test_copy_backward<const int*, random_access_iterator<int*> >();
    test_copy_backward<const int*, int*>();

#if TEST_STD_VER > 17
    test_copy_backward<contiguous_iterator<const int*>, bidirectional_iterator<int*>>();
    test_copy_backward<contiguous_iterator<const int*>, random_access_iterator<int*>>();
    test_copy_backward<contiguous_iterator<const int*>, int*>();

    test_copy_backward<bidirectional_iterator<const int*>, contiguous_iterator<int*>>();
    test_copy_backward<random_access_iterator<const int*>, contiguous_iterator<int*>>();
    test_copy_backward<contiguous_iterator<const int*>, contiguous_iterator<int*>>();
    test_copy_backward<const int*, contiguous_iterator<int*>>();
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
