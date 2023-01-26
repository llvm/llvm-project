//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit zip_view(Views...)

#include <ranges>
#include <tuple>

#include "types.h"

template <class T>
void conversion_test(T);

template <class T, class... Args>
concept implicitly_constructible_from = requires(Args&&... args) { conversion_test<T>({std::move(args)...}); };

// test constructor is explicit
static_assert(std::constructible_from<std::ranges::zip_view<SimpleCommon>, SimpleCommon>);
static_assert(!implicitly_constructible_from<std::ranges::zip_view<SimpleCommon>, SimpleCommon>);

static_assert(std::constructible_from<std::ranges::zip_view<SimpleCommon, SimpleCommon>, SimpleCommon, SimpleCommon>);
static_assert(
    !implicitly_constructible_from<std::ranges::zip_view<SimpleCommon, SimpleCommon>, SimpleCommon, SimpleCommon>);

struct MoveAwareView : std::ranges::view_base {
  int moves = 0;
  constexpr MoveAwareView() = default;
  constexpr MoveAwareView(MoveAwareView&& other) : moves(other.moves + 1) { other.moves = 1; }
  constexpr MoveAwareView& operator=(MoveAwareView&& other) {
    moves = other.moves + 1;
    other.moves = 0;
    return *this;
  }
  constexpr const int* begin() const { return &moves; }
  constexpr const int* end() const { return &moves + 1; }
};

template <class View1, class View2>
constexpr void constructorTest(auto&& buffer1, auto&& buffer2) {
  std::ranges::zip_view v{View1{buffer1}, View2{buffer2}};
  auto [i, j] = *v.begin();
  assert(i == buffer1[0]);
  assert(j == buffer2[0]);
};

constexpr bool test() {

  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int buffer2[4] = {9, 8, 7, 6};

  {
    // constructor from views
    std::ranges::zip_view v(SizedRandomAccessView{buffer}, std::views::iota(0), std::ranges::single_view(2.));
    auto [i, j, k] = *v.begin();
    assert(i == 1);
    assert(j == 0);
    assert(k == 2.0);
  }

  {
    // arguments are moved once
    MoveAwareView mv;
    std::ranges::zip_view v{std::move(mv), MoveAwareView{}};
    auto [numMoves1, numMoves2] = *v.begin();
    assert(numMoves1 == 2); // one move from the local variable to parameter, one move from parameter to member
    assert(numMoves2 == 1);
  }

  // input and forward
  {
    constructorTest<InputCommonView, ForwardSizedView>(buffer, buffer2);
  }

  // bidi and random_access
  {
    constructorTest<BidiCommonView, SizedRandomAccessView>(buffer, buffer2);
  }

  // contiguous
  {
    constructorTest<ContiguousCommonView, ContiguousCommonView>(buffer, buffer2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
