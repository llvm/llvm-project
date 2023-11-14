//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Older Clangs do not support the C++20 feature to constrain destructors
// XFAIL: apple-clang-14

// template<class... Args>
//   constexpr T& emplace(Args&&... args) noexcept;
// Constraints: is_nothrow_constructible_v<T, Args...> is true.
//
// Effects: Equivalent to:
// if (has_value()) {
//   destroy_at(addressof(val));
// } else {
//   destroy_at(addressof(unex));
//   has_val = true;
// }
// return *construct_at(addressof(val), std::forward<Args>(args)...);

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "../../types.h"
#include "test_macros.h"

template <class T, class... Args>
concept CanEmplace = requires(T t, Args&&... args) { t.emplace(std::forward<Args>(args)...); };

static_assert(CanEmplace<std::expected<int, int>, int>);

template <bool Noexcept>
struct CtorFromInt {
  CtorFromInt(int) noexcept(Noexcept);
  CtorFromInt(int, int) noexcept(Noexcept);
};

static_assert(CanEmplace<std::expected<CtorFromInt<true>, int>, int>);
static_assert(CanEmplace<std::expected<CtorFromInt<true>, int>, int, int>);
static_assert(!CanEmplace<std::expected<CtorFromInt<false>, int>, int>);
static_assert(!CanEmplace<std::expected<CtorFromInt<false>, int>, int, int>);

constexpr bool test() {
  // has_value
  {
    BothNoexcept::state oldState{};
    BothNoexcept::state newState{};
    std::expected<BothNoexcept, int> e(std::in_place, oldState, 5);
    decltype(auto) x = e.emplace(newState, 10);
    static_assert(std::same_as<decltype(x), BothNoexcept&>);
    assert(&x == &(*e));

    assert(oldState.dtorCalled);
    assert(e.has_value());
    assert(e.value().data_ == 10);
  }

  // !has_value
  {
    BothMayThrow::state oldState{};
    std::expected<int, BothMayThrow> e(std::unexpect, oldState, 5);
    decltype(auto) x = e.emplace(10);
    static_assert(std::same_as<decltype(x), int&>);
    assert(&x == &(*e));

    assert(oldState.dtorCalled);
    assert(e.has_value());
    assert(e.value() == 10);
  }

  // TailClobberer
  {
    std::expected<TailClobberer<0>, bool> e(std::unexpect);
    e.emplace();
    assert(e.has_value());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
