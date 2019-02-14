//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// template <class _Tp> using __is_inplace_type

#include <utility>

struct S {};

int main(int, char**) {
  using T = std::in_place_type_t<int>;
  static_assert( std::__is_inplace_type<T>::value, "");
  static_assert( std::__is_inplace_type<const T>::value, "");
  static_assert( std::__is_inplace_type<const volatile T>::value, "");
  static_assert( std::__is_inplace_type<T&>::value, "");
  static_assert( std::__is_inplace_type<const T&>::value, "");
  static_assert( std::__is_inplace_type<const volatile T&>::value, "");
  static_assert( std::__is_inplace_type<T&&>::value, "");
  static_assert( std::__is_inplace_type<const T&&>::value, "");
  static_assert( std::__is_inplace_type<const volatile T&&>::value, "");
  static_assert(!std::__is_inplace_type<std::in_place_index_t<0>>::value, "");
  static_assert(!std::__is_inplace_type<std::in_place_t>::value, "");
  static_assert(!std::__is_inplace_type<void>::value, "");
  static_assert(!std::__is_inplace_type<int>::value, "");
  static_assert(!std::__is_inplace_type<S>::value, "");

  return 0;
}
