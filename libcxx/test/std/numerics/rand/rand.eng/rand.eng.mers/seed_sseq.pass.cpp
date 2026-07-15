//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template <class UIntType, size_t w, size_t n, size_t m, size_t r,
//           UIntType a, size_t u, UIntType d, size_t s,
//           UIntType b, size_t t, UIntType c, size_t l, UIntType f>
// class mersenne_twister_engine;

// template<class Sseq> void seed(Sseq& q);

#include <random>
#include <cassert>

#include "test_macros.h"

void
test1()
{
    unsigned a[] = {3, 5, 7};
    std::seed_seq sseq(a, a+3);
    std::mt19937 e1;
    std::mt19937 e2(sseq);
    assert(e1 != e2);
    e1.seed(sseq);
    assert(e1 == e2);
}

void
test2()
{
    unsigned a[] = {3, 5, 7};
    std::seed_seq sseq(a, a+3);
    std::mt19937_64 e1;
    std::mt19937_64 e2(sseq);
    assert(e1 != e2);
    e1.seed(sseq);
    assert(e1 == e2);
}

void test3() {
  unsigned a[] = {15, 22, 288, 37};
  std::seed_seq sseq(a, a + 4);
  std::mt19937 e;
  e.seed(sseq);
  assert(e() == 192527404u);
}

void test4() {
  unsigned a[] = {15, 22, 288, 37};
  std::seed_seq sseq(a, a + 4);
  std::mt19937_64 e;
  e.seed(sseq);
  assert(e() == 11567978440329390872ull);
}

int main(int, char**)
{
    test1();
    test2();
    test3();
    test4();

    return 0;
}
