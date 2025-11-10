//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <utility>

// template<size_t I, class T, T... Values>
//   struct tuple_element<I, integer_sequence<T, Values...>>;
// template<size_t I, class T, T... Values>
//   struct tuple_element<I, const integer_sequence<T, Values...>>;
// template<size_t I, class T, T... Values>
//   constexpr T get(integer_sequence<T, Values...>) noexcept;

#include <cassert>
#include <utility>

void test_structured_bindings() {
  auto [elt0, elt1, elt2, elt3] = std::make_index_sequence<4>();
  
  assert(elt0 == 0);
  assert(elt1 == 1);
  assert(elt2 == 2);
  assert(elt3 == 3);
}

#if __cpp_structured_bindings >= 202411L
template <typename...>
void test_p1061_structured_bindings() {
  auto [... empty] = std::make_index_sequence<0>();
  static_assert(sizeof...(empty) == 0);

  auto [... size4] = std::make_index_sequence<4>();
  static_assert(sizeof...(size4) == 4);

  assert(size4...[0] == 0);
  assert(size4...[1] == 1);
  assert(size4...[2] == 2);
  assert(size4...[3] == 3);
}
#endif

int main(int, char**) {
  test_structured_bindings();
#if __cpp_structured_bindings >= 202411L
  test_p1061_structured_bindings();
#endif
  return 0;
}
