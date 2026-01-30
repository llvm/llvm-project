//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template <class _Tp> using __is_inplace_type

#include <utility>

#include "test_macros.h"

struct S {};

int main(int, char**) {
  using T = std::in_place_type_t<int>;
  static_assert(std::__is_in_place_type_v<T>, "");
  static_assert(std::__is_in_place_type_v<const T>, "");
  static_assert(std::__is_in_place_type_v<const volatile T>, "");
  static_assert(std::__is_in_place_type_v<T&>, "");
  static_assert(std::__is_in_place_type_v<const T&>, "");
  static_assert(std::__is_in_place_type_v<const volatile T&>, "");
  static_assert(std::__is_in_place_type_v<T&&>, "");
  static_assert(std::__is_in_place_type_v<const T&&>, "");
  static_assert(std::__is_in_place_type_v<const volatile T&&>, "");
  static_assert(!std::__is_in_place_type_v<std::in_place_index_t<0>>, "");
  static_assert(!std::__is_in_place_type_v<std::in_place_t>, "");
  static_assert(!std::__is_in_place_type_v<void>, "");
  static_assert(!std::__is_in_place_type_v<int>, "");
  static_assert(!std::__is_in_place_type_v<S>, "");

  return 0;
}
