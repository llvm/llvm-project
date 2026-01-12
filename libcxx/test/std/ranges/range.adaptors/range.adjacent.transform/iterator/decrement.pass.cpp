//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator--() requires bidirectional_range<Base>;
// constexpr iterator operator--(int) requires bidirectional_range<Base>;

#include <array>
#include <cassert>
#include <iterator>
#include <numeric>
#include <ranges>
#include <tuple>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

template <class Iter>
concept canDecrement = requires(Iter it) { --it; } || requires(Iter it) { it--; };

template <class R, class Fn, std::size_t N, class Validator>
constexpr void test_one() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  Validator validator{};

  auto v = R(buffer) | std::views::adjacent_transform<N>(Fn{});

  auto it    = v.begin();
  using Iter = decltype(it);

  std::ranges::advance(it, v.end());

  --it;
  validator(buffer, *it, 9 - N);

  static_assert(std::is_same_v<decltype(--it), Iter&>);
  std::same_as<Iter&> decltype(auto) it_ref = --it;
  assert(&it_ref == &it);

  validator(buffer, *it, 8 - N);

  std::same_as<Iter> decltype(auto) tmp = it--;

  validator(buffer, *tmp, 8 - N);
  validator(buffer, *it, 7 - N);

  // Decrement to the beginning
  for (int i = 6 - N; i >= 0; --i) {
    --it;
    validator(buffer, *it, i);
  }
  assert(it == v.begin());
}

template <std::size_t N, class Fn, class Validator>
constexpr void test() {
  test_one<ContiguousNonCommonSized, Fn, N, Validator>();
  test_one<SimpleCommonRandomAccessSized, Fn, N, Validator>();
  test_one<BidiNonCommonView, Fn, N, Validator>();

  // Non-bidirectional base range
  {
    using View = std::ranges::adjacent_transform_view<ForwardSizedView, Fn, N>;
    using Iter = std::ranges::iterator_t<View>;

    static_assert(!canDecrement<Iter>);
  }
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
