//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// template<class T, size_t Extent>
//   cw-fixed-value(T (&)[Extent]) -> cw-fixed-value<T[Extent]>;                   // exposition only

#include <type_traits>
#include <utility>

constexpr int arr[] = {1, 2, 3};
using T1            = std::constant_wrapper<arr>;
static_assert(std::is_same_v<T1::value_type, const int[3]>);

using T2 = std::constant_wrapper<"hello world">;
static_assert(std::is_same_v<T2::value_type, const char[12]>);

struct S {
  int value;
};

constexpr S s[] = {{1}, {2}, {3}};
using T3        = std::constant_wrapper<s>;
static_assert(std::is_same_v<T3::value_type, const S[3]>);
