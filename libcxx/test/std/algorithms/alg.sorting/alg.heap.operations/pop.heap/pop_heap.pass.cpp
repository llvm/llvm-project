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
//   pop_heap(Iter first, Iter last);

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
    for (int i = N; i > 0; --i)
    {
        std::pop_heap(ia, ia+i);
        assert(std::is_heap(ia, ia+i-1));
    }
    std::pop_heap(ia, ia);
    delete [] ia;
}

int main(int, char**)
{
    test(1000);

  return 0;
}
