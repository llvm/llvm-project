//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr const Pred& pred() const;

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

struct View : std::ranges::view_interface<View> {
  int* begin() const;
  int* end() const;
};

struct Pred {
  int i;
  bool operator()(int) const;
};

constexpr bool test() {
  // &
  {
    std::ranges::drop_while_view<View, Pred> dwv{{}, Pred{5}};
    decltype(auto) x = dwv.pred();
    static_assert(std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  // const &
  {
    const std::ranges::drop_while_view<View, Pred> dwv{{}, Pred{5}};
    decltype(auto) x = dwv.pred();
    static_assert(std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  // &&
  {
    std::ranges::drop_while_view<View, Pred> dwv{{}, Pred{5}};
    decltype(auto) x = std::move(dwv).pred();
    static_assert(std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  // const &&
  {
    const std::ranges::drop_while_view<View, Pred> dwv{{}, Pred{5}};
    decltype(auto) x = std::move(dwv).pred();
    static_assert(std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
