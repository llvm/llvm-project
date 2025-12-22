//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit adjacent_transform_view(View, F)

#include <numeric>
#include <ranges>
#include <tuple>

#include "../range_adaptor_types.h"
#include "helpers.h"

template <class T>
void conversion_test(T);

template <class T, class... Args>
concept implicitly_constructible_from = requires(Args&&... args) { conversion_test<T>({std::move(args)...}); };

// test constructor is explicit
static_assert(
    std::constructible_from<std::ranges::adjacent_transform_view<SimpleCommon, MakeTuple, 1>, SimpleCommon, MakeTuple>);
static_assert(!implicitly_constructible_from<std::ranges::adjacent_transform_view<SimpleCommon, MakeTuple, 1>,
                                             SimpleCommon,
                                             MakeTuple>);

static_assert(std::constructible_from<std::ranges::adjacent_transform_view<SimpleCommon, Tie, 5>, SimpleCommon, Tie>);
static_assert(
    !implicitly_constructible_from<std::ranges::adjacent_transform_view<SimpleCommon, Tie, 5>, SimpleCommon, Tie>);

struct MoveAwareView : std::ranges::view_base {
  int moves                 = 0;
  constexpr MoveAwareView() = default;
  constexpr MoveAwareView(MoveAwareView&& other) : moves(other.moves + 1) { other.moves = 1; }
  constexpr MoveAwareView& operator=(MoveAwareView&& other) {
    moves       = other.moves + 1;
    other.moves = 0;
    return *this;
  }
  constexpr const int* begin() const { return &moves; }
  constexpr const int* end() const { return &moves + 1; }
};

template <class View, class Fn, std::size_t N, class Validator>
constexpr void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  std::ranges::adjacent_transform_view<View, Fn, N> v{View{buffer}, Fn{}};
  Validator validator{};
  auto it = v.begin();
  validator(buffer, *it, 0);
}

template <std::size_t N, class Fn, class Validator>
constexpr void test() {
  {
    // arguments are moved once
    MoveAwareView mv;
    std::ranges::adjacent_transform_view<MoveAwareView, GetFirst, N> v{std::move(mv), GetFirst{}};
    auto& first = *v.begin();
    // one move from the local variable to parameter, one move from parameter to adjacent_view, and one move to adjacent_transform_view
    assert(first == 3);
  }

  test<ForwardSizedView, Fn, N, Validator>();
  test<BidiCommonView, Fn, N, Validator>();
  test<SizedRandomAccessView, Fn, N, Validator>();
  test<ContiguousCommonView, Fn, N, Validator>();
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
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
