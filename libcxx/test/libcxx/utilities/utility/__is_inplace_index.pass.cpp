//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template <class _Tp> using __is_inplace_index

#include <utility>

#include "test_macros.h"

struct S {};

int main(int, char**) {
  using I = std::in_place_index_t<0>;
  static_assert(std::__is_in_place_index_v<I>, "");
  static_assert(std::__is_in_place_index_v<const I>, "");
  static_assert(std::__is_in_place_index_v<const volatile I>, "");
  static_assert(std::__is_in_place_index_v<I&>, "");
  static_assert(std::__is_in_place_index_v<const I&>, "");
  static_assert(std::__is_in_place_index_v<const volatile I&>, "");
  static_assert(std::__is_in_place_index_v<I&&>, "");
  static_assert(std::__is_in_place_index_v<const I&&>, "");
  static_assert(std::__is_in_place_index_v<const volatile I&&>, "");
  static_assert(!std::__is_in_place_index_v<std::in_place_type_t<int>>, "");
  static_assert(!std::__is_in_place_index_v<std::in_place_t>, "");
  static_assert(!std::__is_in_place_index_v<void>, "");
  static_assert(!std::__is_in_place_index_v<int>, "");
  static_assert(!std::__is_in_place_index_v<S>, "");

  return 0;
}
