//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// <functional>

// template<class R, class T> constexpr unspecified mem_fn(R T::*) noexcept;       // constexpr in C++20

#include <functional>
#include <cassert>

struct A
{
    double data_;
};

template <class F>
void
test(F f)
{
    {
    A a;
    f(a) = 5;
    assert(a.data_ == 5);
    A* ap = &a;
    f(ap) = 6;
    assert(a.data_ == 6);
    const A* cap = ap;
    assert(f(cap) == f(ap));
    f(cap) = 7;
    }
}

int main(int, char**)
{
    test(std::mem_fn(&A::data_));

  return 0;
}
