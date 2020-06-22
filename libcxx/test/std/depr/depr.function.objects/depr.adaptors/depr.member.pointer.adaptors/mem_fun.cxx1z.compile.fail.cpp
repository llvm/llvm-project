//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template<Returnable S, ClassType T>
//   mem_fun_t<S,T>
//   mem_fun(S (T::*f)());
// Removed in c++17
// UNSUPPORTED: c++03, c++11, c++14

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <cassert>

#include "test_macros.h"

struct A
{
    char a1() {return 5;}
    short a2(int i) {return short(i+1);}
    int a3() const {return 1;}
    double a4(unsigned i) const {return i-1;}
};

int main(int, char**)
{
    A a;
    assert(std::mem_fun(&A::a1)(&a) == 5);

  return 0;
}
