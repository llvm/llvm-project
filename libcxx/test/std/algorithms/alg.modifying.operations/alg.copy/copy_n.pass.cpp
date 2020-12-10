//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: clang-8

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter   // constexpr after C++17
//   copy_n(InIter first, InIter::difference_type n, OutIter result);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "user_defined_integral.h"

typedef UserDefinedIntegral<unsigned> UDI;

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 void
test_copy_n()
{
    const unsigned N = 1000;
    int ia[N] = {};
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    int ib[N] = {0};

    OutIter r = std::copy_n(InIter(ia), UDI(N/2), OutIter(ib));
    assert(base(r) == ib+N/2);
    for (unsigned i = 0; i < N/2; ++i)
        assert(ia[i] == ib[i]);
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    test_copy_n<input_iterator<const int*>, output_iterator<int*> >();
    test_copy_n<input_iterator<const int*>, input_iterator<int*> >();
    test_copy_n<input_iterator<const int*>, forward_iterator<int*> >();
    test_copy_n<input_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_n<input_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_n<input_iterator<const int*>, int*>();

    test_copy_n<forward_iterator<const int*>, output_iterator<int*> >();
    test_copy_n<forward_iterator<const int*>, input_iterator<int*> >();
    test_copy_n<forward_iterator<const int*>, forward_iterator<int*> >();
    test_copy_n<forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_n<forward_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_n<forward_iterator<const int*>, int*>();

    test_copy_n<bidirectional_iterator<const int*>, output_iterator<int*> >();
    test_copy_n<bidirectional_iterator<const int*>, input_iterator<int*> >();
    test_copy_n<bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test_copy_n<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_n<bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_n<bidirectional_iterator<const int*>, int*>();

    test_copy_n<random_access_iterator<const int*>, output_iterator<int*> >();
    test_copy_n<random_access_iterator<const int*>, input_iterator<int*> >();
    test_copy_n<random_access_iterator<const int*>, forward_iterator<int*> >();
    test_copy_n<random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test_copy_n<random_access_iterator<const int*>, random_access_iterator<int*> >();
    test_copy_n<random_access_iterator<const int*>, int*>();

    test_copy_n<const int*, output_iterator<int*> >();
    test_copy_n<const int*, input_iterator<int*> >();
    test_copy_n<const int*, forward_iterator<int*> >();
    test_copy_n<const int*, bidirectional_iterator<int*> >();
    test_copy_n<const int*, random_access_iterator<int*> >();
    test_copy_n<const int*, int*>();

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
