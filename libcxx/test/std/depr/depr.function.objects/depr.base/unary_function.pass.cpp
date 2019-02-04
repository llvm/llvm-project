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

// template <class Arg, class Result>
// struct unary_function
// {
//     typedef Arg    argument_type;
//     typedef Result result_type;
// };

#include <functional>
#include <type_traits>

int main(int, char**)
{
    static_assert((std::is_same<std::unary_function<unsigned, char>::argument_type, unsigned>::value), "");
    static_assert((std::is_same<std::unary_function<unsigned, char>::result_type, char>::value), "");

  return 0;
}
