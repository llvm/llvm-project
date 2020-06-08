//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// REQUIRES: c++03 || c++11 || c++14
// binary_function was removed in C++17

// binary_function

#include <functional>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::binary_function<int, short, bool> bf;
    static_assert((std::is_same<bf::first_argument_type, int>::value), "");
    static_assert((std::is_same<bf::second_argument_type, short>::value), "");
    static_assert((std::is_same<bf::result_type, bool>::value), "");

  return 0;
}
