//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class UIntType, size_t w, size_t s, size_t r>
// class subtract_with_carry_engine;

// void seed(result_type s = default_seed);

#include <random>
#include <cassert>

void
test1()
{
    for (int s = 0; s < 20; ++s)
    {
        typedef std::ranlux24_base E;
        E e1(s);
        E e2;
        e2.seed(s);
        assert(e1 == e2);
    }
}

void
test2()
{
    for (int s = 0; s < 20; ++s)
    {
        typedef std::ranlux48_base E;
        E e1(s);
        E e2;
        e2.seed(s);
        assert(e1 == e2);
    }
}

int main(int, char**)
{
    test1();
    test2();

  return 0;
}
