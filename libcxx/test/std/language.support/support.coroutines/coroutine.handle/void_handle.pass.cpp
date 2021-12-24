//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-coroutines

#include <coroutine>
#include <type_traits>

#include "test_macros.h"

struct A {
  using promise_type = A*;
};

struct B {};
struct C {};

template <>
struct std::coroutine_traits<A, int> {
  using promise_type = int*;
};
template <class ...Args>
struct std::coroutine_traits<B, Args...> {
  using promise_type = B*;
};
template <>
struct std::coroutine_traits<C> {
  using promise_type = void;
};

template <class Expect, class T, class ...Args>
void check_type() {
  using P = typename std::coroutine_traits<T, Args...>::promise_type ;
  static_assert(std::is_same<P, Expect>::value, "");
};

int main(int, char**)
{
  check_type<A*, A>();
  check_type<int*, A, int>();
  check_type<B*, B>();
  check_type<void, C>();

  return 0;
}
