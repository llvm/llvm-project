// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Make sure that we don't blow up the template instantiation recursion depth
// for tuples of size <= 1024.

#include <tuple>
#include <cassert>
#include <utility>

template <size_t... I>
constexpr void CreateTuple(std::index_sequence<I...>) {
  std::tuple<decltype(I)...> tuple(I...);
  assert(std::get<0>(tuple) == 0);
  assert(std::get<sizeof...(I)-1>(tuple) == sizeof...(I)-1);
}

constexpr bool test() {
  CreateTuple(std::make_index_sequence<1024>{});
  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");
  return 0;
}
