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
concept HasFreeEmpty = requires(R r) { std::ranges::empty(r); };

template <typename R>
concept HasMemberEmpty = requires(R r) {
  { r.empty() } -> std::same_as<bool>;
};

constexpr void test_empty_iota() {
  std::vector<int> ev;

  // Both parameters are non-const
  {
    auto iv = std::views::iota(std::ranges::begin(ev), std::ranges::end(ev));

    static_assert(HasFreeEmpty<decltype(iv)>);
    static_assert(HasMemberEmpty<decltype(iv)>);

    assert(iv.empty());
  }
  // Left paramter is const
  {
    auto iv = std::views::iota(std::ranges::begin(std::as_const(ev)), std::ranges::end(ev));

    static_assert(HasFreeEmpty<decltype(iv)>);
    static_assert(HasMemberEmpty<decltype(iv)>);

    assert(iv.empty());
  }
  // Right paramter is const
  {
    auto iv = std::views::iota(std::ranges::begin(ev), std::ranges::end(std::as_const(ev)));

    static_assert(HasFreeEmpty<decltype(iv)>);
    static_assert(HasMemberEmpty<decltype(iv)>);

    assert(iv.empty());
  }
  // Both parameters are const
  {
    auto iv = std::views::iota(std::ranges::begin(std::as_const(ev)), std::ranges::end(std::as_const(ev)));

    static_assert(HasFreeEmpty<decltype(iv)>);
    static_assert(HasMemberEmpty<decltype(iv)>);

    assert(iv.empty());
  }

  std::vector<char> v{'b', 'a', 'b', 'a', 'z', 'm', 't'};
  auto fv = v | std::views::filter([](auto val) { return val == '0'; });

  {
    auto iv = std::views::iota(std::ranges::begin(fv), std::ranges::end(fv));

    static_assert(HasFreeEmpty<decltype(iv)>);
    static_assert(HasMemberEmpty<decltype(iv)>);

    assert(iv.empty());
  }
}

constexpr void test_nonempty_iota() {
  // Default ctr
  {
    std::ranges::iota_view<Int42<DefaultTo42>> iv;

    static_assert(HasFreeEmpty<decltype(iv)>);
    static_assert(HasMemberEmpty<decltype(iv)>);

    assert(!iv.empty());
  }
  // Value pass
  {
    std::ranges::iota_view<SomeInt> iv(SomeInt(94));

    static_assert(HasFreeEmpty<decltype(iv)>);
    static_assert(HasMemberEmpty<decltype(iv)>);

    assert(!iv.empty());
  }

  {
    std::vector<char> v;
    auto it = std::back_inserter(v);
    auto iv = std::views::iota(it);

    static_assert(HasFreeEmpty<decltype(iv)>);
    static_assert(HasMemberEmpty<decltype(iv)>);

    assert(!iv.empty());
  }
  {
    std::vector<char> v{'b', 'a', 'b', 'a', 'z', 'm', 't'};
    auto it = std::back_inserter(v);
    auto iv = std::views::iota(it);

    static_assert(HasFreeEmpty<decltype(iv)>);
    static_assert(HasMemberEmpty<decltype(iv)>);

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
