//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// REQUIRES: c++98 || c++03 || c++11 || c++14

// pointer_to_binary_function

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

double binary_f(int i, short j) {return i - j + .75;}

int main(int, char**)
{
    typedef std::pointer_to_binary_function<int, short, double> F;
    static_assert((std::is_base_of<std::binary_function<int, short, double>, F>::value), "");
    const F f(binary_f);
    assert(f(36, 27) == 9.75);

  return 0;
}
