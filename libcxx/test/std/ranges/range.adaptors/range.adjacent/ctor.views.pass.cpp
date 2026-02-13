//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit adjacent_view(View)

#include <ranges>
#include <tuple>

#include "../range_adaptor_types.h"

template <class T>
void conversion_test(T);

template <class T, class... Args>
concept implicitly_constructible_from = requires(Args&&... args) { conversion_test<T>({std::move(args)...}); };

// test constructor is explicit
static_assert(std::constructible_from<std::ranges::adjacent_view<SimpleCommon, 1>, SimpleCommon>);
static_assert(!implicitly_constructible_from<std::ranges::adjacent_view<SimpleCommon, 1>, SimpleCommon>);

static_assert(std::constructible_from<std::ranges::adjacent_view<SimpleCommon, 5>, SimpleCommon>);
static_assert(!implicitly_constructible_from<std::ranges::adjacent_view<SimpleCommon, 5>, SimpleCommon>);

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

template <class View, std::size_t N>
constexpr void constructor_test(auto&& buffer) {
  std::ranges::adjacent_view<View, N> v{View{buffer}};
  auto tuple = *v.begin();
  assert(std::get<0>(tuple) == buffer[0]);
  if constexpr (N >= 2)
    assert(std::get<1>(tuple) == buffer[1]);
  if constexpr (N >= 3)
    assert(std::get<2>(tuple) == buffer[2]);
};

template <std::size_t N>
constexpr void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // arguments are moved once
    MoveAwareView mv;
    std::ranges::adjacent_view<MoveAwareView, 1> v{std::move(mv)};
    auto tuple = *v.begin();
    assert(std::get<0>(tuple) == 2); // one move from the local variable to parameter, one move from parameter to member
  }

  // forward
  {
    constructor_test<ForwardSizedView, N>(buffer);
  }

  // bidi
  {
    constructor_test<BidiCommonView, N>(buffer);
  }

  // random_access
  {
    constructor_test<SizedRandomAccessView, N>(buffer);
  }

  // contiguous
  {
    constructor_test<ContiguousCommonView, N>(buffer);
  }
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
