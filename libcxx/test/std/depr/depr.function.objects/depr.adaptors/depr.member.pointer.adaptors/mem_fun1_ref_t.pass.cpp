//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// REQUIRES: c++98 || c++03 || c++11 || c++14

// mem_fun1_ref_t

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <type_traits>
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
    typedef std::mem_fun1_ref_t<short, A, int> F;
    static_assert((std::is_base_of<std::binary_function<A, int, short>, F>::value), "");
    const F f(&A::a2);
    A a;
    assert(f(a, 5) == 6);

  return 0;
}
