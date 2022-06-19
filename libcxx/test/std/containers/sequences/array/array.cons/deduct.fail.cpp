//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>
// UNSUPPORTED: c++03, c++11, c++14

// template <class T, class... U>
//   array(T, U...) -> array<T, 1 + sizeof...(U)>;
//
//  Requires: (is_same_v<T, U> && ...) is true. Otherwise the program is ill-formed.


#include <array>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

int main(int, char**)
{
    {
    std::array arr{1,2,3L}; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'array'}}
    }

  return 0;
}
