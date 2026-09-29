//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator++();
// constexpr iterator operator++(int);

#include <array>
#include <cassert>
#include <concepts>
#include <numeric>
#include <ranges>
#include <tuple>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

template <class R, class Fn, std::size_t N, class Validator>
constexpr void test() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

  Validator validator{};

  auto v     = R(buffer) | std::views::adjacent_transform<N>(Fn{});
  auto it    = v.begin();
  using Iter = decltype(it);

  validator(buffer, *it, 0);

  std::same_as<Iter&> decltype(auto) it_ref = ++it;
  assert(&it_ref == &it);

  validator(buffer, *it, 1);

  static_assert(std::is_same_v<decltype(it++), Iter>);
  auto original                          = it;
  std::same_as<Iter> decltype(auto) copy = it++;
  assert(original == copy);

  validator(buffer, *copy, 1);
  validator(buffer, *it, 2);

  // Increment to the end
  for (std::size_t i = 3; i != 9 - N; ++i) {
    ++it;
    validator(buffer, *it, i);
  }

  ++it;
  assert(it == v.end());
}

template <std::size_t N, class Fn, class Validator>
constexpr void test() {
  test<ContiguousCommonView, Fn, N, Validator>();
  test<SimpleCommonRandomAccessSized, Fn, N, Validator>();
  test<BidiNonCommonView, Fn, N, Validator>();
  test<ForwardSizedView, Fn, N, Validator>();
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple, ValidateTupleFromIndex<N>>();
  test<N, Tie, ValidateTieFromIndex<N>>();
  test<N, GetFirst, ValidateGetFirstFromIndex<N>>();
  test<N, Multiply, ValidateMultiplyFromIndex<N>>();
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
