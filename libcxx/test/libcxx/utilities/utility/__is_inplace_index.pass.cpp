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
  static_assert( std::__is_inplace_index<I>::value, "");
  static_assert( std::__is_inplace_index<const I>::value, "");
  static_assert( std::__is_inplace_index<const volatile I>::value, "");
  static_assert( std::__is_inplace_index<I&>::value, "");
  static_assert( std::__is_inplace_index<const I&>::value, "");
  static_assert( std::__is_inplace_index<const volatile I&>::value, "");
  static_assert( std::__is_inplace_index<I&&>::value, "");
  static_assert( std::__is_inplace_index<const I&&>::value, "");
  static_assert( std::__is_inplace_index<const volatile I&&>::value, "");
  static_assert(!std::__is_inplace_index<std::in_place_type_t<int>>::value, "");
  static_assert(!std::__is_inplace_index<std::in_place_t>::value, "");
  static_assert(!std::__is_inplace_index<void>::value, "");
  static_assert(!std::__is_inplace_index<int>::value, "");
  static_assert(!std::__is_inplace_index<S>::value, "");

  return 0;
}
