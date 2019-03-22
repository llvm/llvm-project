//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <span>

// tuple_element<I, span<T, N> >::type

#include <span>

#include "test_macros.h"


int main(int, char**)
{
//  No tuple_element for dynamic spans
    using T1 = typename std::tuple_element< 0, std::span<int,  0>>::type; // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Index out of bounds in std::tuple_element<> (std::span)"}}
    using T2 = typename std::tuple_element< 5, std::span<int,  5>>::type; // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Index out of bounds in std::tuple_element<> (std::span)"}}
    using T3 = typename std::tuple_element<20, std::span<int, 10>>::type; // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Index out of bounds in std::tuple_element<> (std::span)"}}

    return 0;
}
