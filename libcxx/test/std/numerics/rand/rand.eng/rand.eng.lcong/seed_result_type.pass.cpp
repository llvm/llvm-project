//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template <class UIntType, UIntType a, UIntType c, UIntType m>
//   class linear_congruential_engine;

// void seed(result_type s = default_seed);

#include <random>
#include <cassert>

template <class T>
void
test1()
{
    for (T s = 0; s < 20; ++s)
    {
        typedef std::linear_congruential_engine<T, 2, 3, 7> E;
        E e1(s);
        E e2;
        e2.seed(s);
        assert(e1 == e2);
    }
}

int main()
{
    test1<unsigned short>();
    test1<unsigned int>();
    test1<unsigned long>();
    test1<unsigned long long>();
}
