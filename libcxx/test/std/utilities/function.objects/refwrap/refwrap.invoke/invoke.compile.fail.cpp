//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <class... ArgTypes>
//   requires Callable<T, ArgTypes&&...>
//   Callable<T, ArgTypes&&...>::result_type
//   operator()(ArgTypes&&... args) const;

#include <functional>
#include <cassert>

// member data pointer:  cv qualifiers should transfer from argument to return type

struct A_int_1
{
    A_int_1() : data_(5) {}

    int data_;
};

void
test_int_1()
{
    // member data pointer
    {
    int A_int_1::*fp = &A_int_1::data_;
    std::reference_wrapper<int A_int_1::*> r1(fp);
    A_int_1 a;
    assert(r1(a) == 5);
    r1(a) = 6;
    assert(r1(a) == 6);
    const A_int_1* ap = &a;
    assert(r1(ap) == 6);
    r1(ap) = 7;
    assert(r1(ap) == 7);
    }
}

int main(int, char**)
{
    test_int_1();

  return 0;
}
