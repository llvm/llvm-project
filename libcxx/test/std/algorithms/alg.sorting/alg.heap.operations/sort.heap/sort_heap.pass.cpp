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
//   sort_heap(Iter first, Iter last);

#include <algorithm>
#include <random>
#include <cassert>

std::mt19937 randomness;

void test(int N)
{
    int* ia = new int [N];
    for (int i = 0; i < N; ++i)
        ia[i] = i;
    std::shuffle(ia, ia+N, randomness);
    std::make_heap(ia, ia+N);
    std::sort_heap(ia, ia+N);
    assert(std::is_sorted(ia, ia+N));
    delete [] ia;
}

int main()
{
    test(0);
    test(1);
    test(2);
    test(3);
    test(10);
    test(1000);
}
