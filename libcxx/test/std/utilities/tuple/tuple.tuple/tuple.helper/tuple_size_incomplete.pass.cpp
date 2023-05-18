//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   struct tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// UNSUPPORTED: c++03

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_macros.h"

template <class T, std::size_t Size = sizeof(std::tuple_size<T>)>
constexpr bool is_complete(int) { static_assert(Size > 0, ""); return true; }
template <class> constexpr bool is_complete(long) { return false; }
template <class T> constexpr bool is_complete() { return is_complete<T>(0); }

struct Dummy1 {};
struct Dummy2 {};

namespace std {
template <> struct tuple_size<Dummy1> : public integral_constant<std::size_t, 0> {};
}

template <class T>
void test_complete() {
  static_assert(is_complete<T>(), "");
  static_assert(is_complete<const T>(), "");
  static_assert(is_complete<volatile T>(), "");
  static_assert(is_complete<const volatile T>(), "");
}

template <class T>
void test_incomplete() {
  static_assert(!is_complete<T>(), "");
  static_assert(!is_complete<const T>(), "");
  static_assert(!is_complete<volatile T>(), "");
  static_assert(!is_complete<const volatile T>(), "");
}


int main(int, char**)
{
  test_complete<std::tuple<> >();
  test_complete<std::tuple<int&> >();
  test_complete<std::tuple<int&&, int&, void*>>();
  test_complete<std::pair<int, long> >();
  test_complete<std::array<int, 5> >();
  test_complete<Dummy1>();

  test_incomplete<void>();
  test_incomplete<int>();
  test_incomplete<std::tuple<int>&>();
  test_incomplete<Dummy2>();

  return 0;
}
