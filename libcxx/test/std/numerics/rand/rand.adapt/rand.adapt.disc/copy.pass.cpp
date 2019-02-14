//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class Engine, size_t p, size_t r>
// class discard_block_engine

// discard_block_engine(const discard_block_engine&);

#include <random>
#include <cassert>

void
test1()
{
    typedef std::ranlux24 E;
    E e1;
    (void)e1();
    E e2 = e1;
    assert(e1 == e2);
    assert(e1() == e2());
    E::result_type k = e1();
    assert(e1 != e2);
    assert(e2() == k);
    assert(e1 == e2);
}

void
test2()
{
    typedef std::ranlux48 E;
    E e1;
    (void)e1();
    E e2 = e1;
    assert(e1 == e2);
    assert(e1() == e2());
    E::result_type k = e1();
    assert(e1 != e2);
    assert(e2() == k);
    assert(e1 == e2);
}

int main(int, char**)
{
    test1();
    test2();

  return 0;
}
