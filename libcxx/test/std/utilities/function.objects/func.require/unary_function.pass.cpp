//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// REQUIRES: c++98 || c++03 || c++11 || c++14
// unary_function was removed in C++17

// unary_function

#include <functional>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::unary_function<int, bool> uf;
    static_assert((std::is_same<uf::argument_type, int>::value), "");
    static_assert((std::is_same<uf::result_type, bool>::value), "");

  return 0;
}
