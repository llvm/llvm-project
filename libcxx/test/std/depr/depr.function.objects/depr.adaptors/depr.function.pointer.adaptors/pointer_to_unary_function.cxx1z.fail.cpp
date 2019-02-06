//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// pointer_to_unary_function
// UNSUPPORTED: c++98, c++03, c++11, c++14

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

double unary_f(int i) {return 0.5 - i;}

int main()
{
    typedef std::pointer_to_unary_function<int, double> F;
}
