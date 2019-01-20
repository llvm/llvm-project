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

// void discard(unsigned long long z);

#include <random>
#include <cassert>

void
test1()
{
    std::ranlux24_base e1;
    std::ranlux24_base e2 = e1;
    assert(e1 == e2);
    e1.discard(3);
    assert(e1 != e2);
    (void)e2();
    (void)e2();
    (void)e2();
    assert(e1 == e2);
}

void
test2()
{
    std::ranlux48_base e1;
    std::ranlux48_base e2 = e1;
    assert(e1 == e2);
    e1.discard(3);
    assert(e1 != e2);
    (void)e2();
    (void)e2();
    (void)e2();
    assert(e1 == e2);
}

int main()
{
    test1();
    test2();
}
