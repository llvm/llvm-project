//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Make sure that we don't blow up the template instantiation recursion depth
// for tuples of size <= 512.

#include <tuple>
#include <cassert>
#include <utility>

#include "test_macros.h"

template <std::size_t... I>
constexpr void CreateTuple(std::index_sequence<I...>) {
  using LargeTuple = std::tuple<std::integral_constant<std::size_t, I>...>;
  using TargetTuple = std::tuple<decltype(I)...>;
  LargeTuple tuple(std::integral_constant<std::size_t, I>{}...);
  assert(std::get<0>(tuple).value == 0);
  assert(std::get<sizeof...(I)-1>(tuple).value == sizeof...(I)-1);

  TargetTuple t1 = tuple;                                  // converting copy constructor from &
  TargetTuple t2 = static_cast<LargeTuple const&>(tuple);  // converting copy constructor from const&
  TargetTuple t3 = std::move(tuple);                       // converting rvalue constructor
  TargetTuple t4 = static_cast<LargeTuple const&&>(tuple); // converting const rvalue constructor
  TargetTuple t5;                                          // default constructor
  (void)t1; (void)t2; (void)t3; (void)t4; (void)t5;

#if TEST_STD_VER >= 20
  t1 = tuple;                                              // converting assignment from &
  t1 = static_cast<LargeTuple const&>(tuple);              // converting assignment from const&
  t1 = std::move(tuple);                                   // converting assignment from &&
  t1 = static_cast<LargeTuple const&&>(tuple);             // converting assignment from const&&
  swap(t1, t2);                                            // swap
#endif
  // t1 == tuple;                                          // comparison does not work yet (we blow the constexpr stack)
}

constexpr bool test() {
  CreateTuple(std::make_index_sequence<512>{});
  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");
  return 0;
}
