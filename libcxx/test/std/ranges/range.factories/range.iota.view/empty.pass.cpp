//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr bool empty() const;

#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>
#include <vector>

#include "types.h"

template <typename R>
concept HasEmpty = requires(const R r) {
  std::ranges::empty(r);
  { r.empty() } -> std::same_as<bool>;
};

constexpr void test_empty_iota_sfinae() {
  std::vector<int> ev;

  auto iv = std::views::iota(std::ranges::begin(ev), std::ranges::end(ev));

  static_assert(HasEmpty<decltype(iv)>);
  static_assert(HasEmpty<decltype(std::as_const(iv))>);
}

constexpr void test_nonempty_iota_sfinae() {
  // Default ctr
  {
    std::ranges::iota_view<Int42<DefaultTo42>> iv;

    static_assert(HasEmpty<decltype(iv)>);
  }
  // Value pass
  {
    std::ranges::iota_view<SomeInt> iv(SomeInt(94));

    static_assert(HasEmpty<decltype(iv)>);
  }

  {
    std::vector<char> v;
    auto it = std::back_inserter(v);
    auto iv = std::views::iota(it);

    static_assert(HasEmpty<decltype(iv)>);
  }
  {
    std::vector<char> v{'b', 'a', 'b', 'a', 'z', 'm', 't'};
    auto it = std::back_inserter(v);
    auto iv = std::views::iota(it);

    static_assert(HasEmpty<decltype(iv)>);
  }
}

constexpr void test_empty_iota() {
  std::vector<int> ev;

  auto iv = std::views::iota(std::ranges::begin(ev), std::ranges::end(ev));

  assert(iv.empty());
  assert(std::as_const(iv).empty());
}

constexpr void test_nonempty_iota() {
  // Default ctr
  {
    std::ranges::iota_view<Int42<DefaultTo42>> iv;

    assert(!iv.empty());
  }
  // Value pass
  {
    std::ranges::iota_view<SomeInt> iv(SomeInt(94));

    assert(!iv.empty());
  }

  {
    std::vector<char> v;
    auto it = std::back_inserter(v);
    auto iv = std::views::iota(it);

    assert(!iv.empty());
  }
  {
    std::vector<char> v{'b', 'a', 'b', 'a', 'z', 'm', 't'};
    auto it = std::back_inserter(v);
    auto iv = std::views::iota(it);

    assert(!iv.empty());
  }
}

constexpr bool test() {
  test_empty_iota();
  test_nonempty_iota();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
