//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr explicit cartesian_product_view(First first_base, Vs... bases);

#include <cassert>
#include <ranges>
#include <tuple>
#include <utility>

#include "../range_adaptor_types.h"

template <class T>
void conversion_test(T);

template <class T, class... Args>
concept implicitly_constructible_from = requires(Args&&... args) { conversion_test<T>({std::move(args)...}); };

// The constructor is explicit (cannot be invoked through copy-list-initialisation).
static_assert(std::constructible_from<std::ranges::cartesian_product_view<SimpleCommon>, SimpleCommon>);
static_assert(!implicitly_constructible_from<std::ranges::cartesian_product_view<SimpleCommon>, SimpleCommon>);

static_assert(std::constructible_from<std::ranges::cartesian_product_view<SimpleCommon, SimpleCommon>,
                                      SimpleCommon,
                                      SimpleCommon>);
static_assert(!implicitly_constructible_from<std::ranges::cartesian_product_view<SimpleCommon, SimpleCommon>,
                                             SimpleCommon,
                                             SimpleCommon>);

// 4-range constructor is also explicit.
static_assert(std::constructible_from<
              std::ranges::cartesian_product_view<SimpleCommon, SimpleCommon, SimpleCommon, SimpleCommon>,
              SimpleCommon,
              SimpleCommon,
              SimpleCommon,
              SimpleCommon>);

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

template <class View1, class View2>
constexpr void constructorTest(auto&& buffer1, auto&& buffer2) {
  std::ranges::cartesian_product_view v{View1{buffer1}, View2{buffer2}};
  auto [i, j] = *v.begin();
  assert(i == buffer1[0]);
  assert(j == buffer2[0]);
}

constexpr bool test() {
  int buffer[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  int buffer2[] = {9, 8, 7, 6};

  { // construction from views
    std::ranges::cartesian_product_view v(
        SizedRandomAccessView{buffer}, std::views::iota(0, 5), std::ranges::single_view(2.));
    assert(*v.begin() == std::make_tuple(1, 0, 2.0));
  }

  { // 4-range construction
    std::ranges::cartesian_product_view v(
        SizedRandomAccessView{buffer}, std::views::iota(0, 2), std::ranges::single_view(2.), std::views::iota(10, 13));
    assert(v.size() == 8u * 2u * 1u * 3u);
    auto [a, b, c, d] = *v.begin();
    assert(a == 1 && b == 0 && c == 2.0 && d == 10);
  }

  { // each argument is moved exactly once into the bases tuple
    MoveAwareView mv;
    std::ranges::cartesian_product_view v{std::move(mv), MoveAwareView{}};
    auto [numMoves1, numMoves2] = *v.begin();
    assert(numMoves1 == 2); // local -> parameter, parameter -> member
    assert(numMoves2 == 1);
  }

  constructorTest<InputCommonView, ForwardSizedView>(buffer, buffer2);
  constructorTest<BidiCommonView, SizedRandomAccessView>(buffer, buffer2);
  constructorTest<ContiguousCommonView, ContiguousCommonView>(buffer, buffer2);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
