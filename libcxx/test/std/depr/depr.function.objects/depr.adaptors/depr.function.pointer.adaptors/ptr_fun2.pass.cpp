//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// REQUIRES: c++03 || c++11 || c++14

// template <CopyConstructible Arg1, CopyConstructible Arg2, Returnable Result>
// pointer_to_binary_function<Arg1,Arg2,Result>
// ptr_fun(Result (*f)(Arg1, Arg2));

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

double binary_f(int i, short j) {return i - j + .75;}

int main(int, char**)
{
    assert(std::ptr_fun(binary_f)(36, 27) == 9.75);

  return 0;
}
