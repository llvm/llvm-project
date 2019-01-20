//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// REQUIRES: c++98 || c++03 || c++11 || c++14

// pointer_to_unary_function

#include <functional>
#include <type_traits>
#include <cassert>

double unary_f(int i) {return 0.5 - i;}

int main()
{
    typedef std::pointer_to_unary_function<int, double> F;
    static_assert((std::is_base_of<std::unary_function<int, double>, F>::value), "");
    const F f(unary_f);
    assert(f(36) == -35.5);
}
