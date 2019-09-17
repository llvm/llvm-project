//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter> && LessThanComparable<Iter::value_type>
//   void
//   make_heap(Iter first, Iter last);

#include <algorithm>
#include <random>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

std::mt19937 randomness;

void test(int N)
{
    int* ia = new int [N];
    for (int i = 0; i < N; ++i)
        ia[i] = i;
    std::shuffle(ia, ia+N, randomness);
    std::make_heap(ia, ia+N);
    assert(std::is_heap(ia, ia+N));

    typedef random_access_iterator<int *> RI;
    std::shuffle(RI(ia), RI(ia+N), randomness);
    std::make_heap(RI(ia), RI(ia+N));
    assert(std::is_heap(RI(ia), RI(ia+N)));

    delete [] ia;
}

int main(int, char**)
{
    test(0);
    test(1);
    test(2);
    test(3);
    test(10);
    test(1000);

  return 0;
}
