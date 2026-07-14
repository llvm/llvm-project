//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <algorithm>

#include <__type_traits/desugars_to.h>

#include <algorithm>

// check that __less<> desugars to __totally_ordered_less_tag for integral types regardless of their cv-ref
static_assert(std::__desugars_to_v<std::__totally_ordered_less_tag, std::__less<>, int, int>);
static_assert(std::__desugars_to_v<std::__totally_ordered_less_tag, std::__less<>, int&, int&>);
static_assert(std::__desugars_to_v<std::__totally_ordered_less_tag, std::__less<>, const int, const int>);
static_assert(std::__desugars_to_v<std::__totally_ordered_less_tag, std::__less<>, const int&, const int&>);
static_assert(std::__desugars_to_v<std::__totally_ordered_less_tag, std::__less<>, volatile int&, volatile int&>);
static_assert(
    std::__desugars_to_v<std::__totally_ordered_less_tag, std::__less<>, volatile const int&, volatile const int&>);
static_assert(
    !std::
        __desugars_to_v<std::__totally_ordered_less_tag, std::__less<>, volatile const float&, volatile const float&>);

int main(int, char**) { return 0; }
