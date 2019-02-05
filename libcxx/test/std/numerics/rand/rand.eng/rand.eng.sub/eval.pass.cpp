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

// result_type operator()();

#include <random>
#include <cassert>

void
test1()
{
    std::ranlux24_base e;
    assert(e() == 15039276u);
    assert(e() == 16323925u);
    assert(e() == 14283486u);
}

void
test2()
{
    std::ranlux48_base e;
    assert(e() == 23459059301164ull);
    assert(e() == 28639057539807ull);
    assert(e() == 276846226770426ull);
}

int main(int, char**)
{
    test1();
    test2();

  return 0;
}
