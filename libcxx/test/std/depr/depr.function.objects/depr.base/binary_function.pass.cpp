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

// template <class Arg1, class Arg2, class Result>
// struct binary_function
// {
//     typedef Arg1   first_argument_type;
//     typedef Arg2   second_argument_type;
//     typedef Result result_type;
// };

#include <functional>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<std::binary_function<int, unsigned, char>::first_argument_type, int>::value), "");
    static_assert((std::is_same<std::binary_function<int, unsigned, char>::second_argument_type, unsigned>::value), "");
    static_assert((std::is_same<std::binary_function<int, unsigned, char>::result_type, char>::value), "");

  return 0;
}
