//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class U, class... Args>
//   constexpr T& emplace(initializer_list<U> il, Args&&... args) noexcept;
// Constraints: is_nothrow_constructible_v<T, initializer_list<U>&, Args...> is true.
//
// Effects: Equivalent to:
// if (has_value()) {
//   destroy_at(addressof(val));
// } else {
//   destroy_at(addressof(unex));
//   has_val = true;
// }
// return *construct_at(addressof(val), il, std::forward<Args>(args)...);

#include <algorithm>
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
struct CtorFromInitializerList {
  CtorFromInitializerList(std::initializer_list<int>&) noexcept(Noexcept);
  CtorFromInitializerList(std::initializer_list<int>&, int) noexcept(Noexcept);
};

static_assert(CanEmplace<std::expected<CtorFromInitializerList<true>, int>, std::initializer_list<int>&>);
static_assert(!CanEmplace<std::expected<CtorFromInitializerList<false>, int>, std::initializer_list<int>&>);
static_assert(CanEmplace<std::expected<CtorFromInitializerList<true>, int>, std::initializer_list<int>&, int>);
static_assert(!CanEmplace<std::expected<CtorFromInitializerList<false>, int>, std::initializer_list<int>&, int>);

struct Data {
  std::initializer_list<int> il;
  int i;

  constexpr Data(std::initializer_list<int>& l, int ii) noexcept : il(l), i(ii) {}
};

constexpr bool test() {
  // has_value
  {
    auto list1 = {1, 2, 3};
    auto list2 = {4, 5, 6};
    std::expected<Data, int> e(std::in_place, list1, 5);
    decltype(auto) x = e.emplace(list2, 10);
    static_assert(std::same_as<decltype(x), Data&>);
    assert(&x == &(*e));

    assert(e.has_value());
    assert(std::ranges::equal(e.value().il, list2));
    assert(e.value().i == 10);
  }

  // !has_value
  {
    auto list = {4, 5, 6};
    std::expected<Data, int> e(std::unexpect, 5);
    decltype(auto) x = e.emplace(list, 10);
    static_assert(std::same_as<decltype(x), Data&>);
    assert(&x == &(*e));

    assert(e.has_value());
    assert(std::ranges::equal(e.value().il, list));
    assert(e.value().i == 10);
  }

  // TailClobberer
  {
    std::expected<TailClobberer<0>, bool> e(std::unexpect);
    auto list = {4, 5, 6};
    e.emplace(list);
    assert(e.has_value());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
