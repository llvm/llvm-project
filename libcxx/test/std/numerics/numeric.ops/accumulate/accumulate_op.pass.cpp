//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// template <InputIterator Iter, MoveConstructible T,
//           Callable<auto, const T&, Iter::reference> BinaryOperation>
//   requires HasAssign<T, BinaryOperation::result_type>
//         && CopyConstructible<BinaryOperation>
//   T
//   accumulate(Iter first, Iter last, T init, BinaryOperation binary_op);

#include <numeric>
#include <functional>
#include <cassert>

#include "test_iterators.h"

template <class Iter, class T>
void
test(Iter first, Iter last, T init, T x)
{
    assert(std::accumulate(first, last, init, std::multiplies<T>()) == x);
}

template <class Iter>
void
test()
{
    int ia[] = {1, 2, 3, 4, 5, 6};
    unsigned sa = sizeof(ia) / sizeof(ia[0]);
    test(Iter(ia), Iter(ia), 1, 1);
    test(Iter(ia), Iter(ia), 10, 10);
    test(Iter(ia), Iter(ia+1), 1, 1);
    test(Iter(ia), Iter(ia+1), 10, 10);
    test(Iter(ia), Iter(ia+2), 1, 2);
    test(Iter(ia), Iter(ia+2), 10, 20);
    test(Iter(ia), Iter(ia+sa), 1, 720);
    test(Iter(ia), Iter(ia+sa), 10, 7200);
}

int main(int, char**)
{
    test<input_iterator<const int*> >();
    test<forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();

  return 0;
}
