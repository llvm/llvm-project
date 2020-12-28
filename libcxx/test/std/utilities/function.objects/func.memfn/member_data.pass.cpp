//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template<Returnable R, class T> unspecified mem_fn(R T::* pm);

#include <functional>
#include <cassert>

#include "test_macros.h"

struct A
{
    double data_;
};

template <class F>
TEST_CONSTEXPR_CXX20 bool
test(F f)
{
    {
    A a = {0.0};
    f(a) = 5;
    assert(a.data_ == 5);
    A* ap = &a;
    f(ap) = 6;
    assert(a.data_ == 6);
    const A* cap = ap;
    assert(f(cap) == f(ap));
    const F& cf = f;
    assert(cf(ap) == f(ap));
    }
    return true;
}

int main(int, char**)
{
    test(std::mem_fn(&A::data_));

#if TEST_STD_VER >= 20
    static_assert(test(std::mem_fn(&A::data_)));
#endif

    return 0;
}
