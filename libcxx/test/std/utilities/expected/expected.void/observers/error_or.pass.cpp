//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class G = E> constexpr E error_or(G&& e) const &;
// template<class G = E> constexpr E error_or(G&& e) &&;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

struct ConstructFromInt {
  int value;
  constexpr ConstructFromInt(int v) : value(v) {}
};

constexpr bool test_default_template_arg() {
  // const &, has_value()
  {
    const std::expected<void, ConstructFromInt> e;
    std::same_as<ConstructFromInt> decltype(auto) x = e.error_or(10);
    assert(x.value == 10);
  }

  // const &, !has_value()
  {
    const std::expected<void, ConstructFromInt> e(std::unexpect, 5);
    std::same_as<ConstructFromInt> decltype(auto) x = e.error_or(10);
    assert(x.value == 5);
  }

  // &&, has_value()
  {
    const std::expected<void, ConstructFromInt> e;
    std::same_as<ConstructFromInt> decltype(auto) x = std::move(e).error_or(10);
    assert(x.value == 10);
  }

  // &&, !has_value()
  {
    const std::expected<void, ConstructFromInt> e(std::unexpect, 5);
    std::same_as<ConstructFromInt> decltype(auto) x = std::move(e).error_or(10);
    assert(x.value == 5);
  }

  return true;
}

constexpr bool test() {
  // const &, has_value()
  {
    const std::expected<void, int> e;
    std::same_as<int> decltype(auto) x = e.error_or(10);
    assert(x == 10);
  }

  // const &, !has_value()
  {
    const std::expected<void, int> e(std::unexpect, 5);
    std::same_as<int> decltype(auto) x = e.error_or(10);
    assert(x == 5);
  }

  // &&, has_value()
  {
    std::expected<void, int> e;
    std::same_as<int> decltype(auto) x = std::move(e).error_or(10);
    assert(x == 10);
  }

  // &&, !has_value()
  {
    std::expected<void, int> e(std::unexpect, 5);
    std::same_as<int> decltype(auto) x = std::move(e).error_or(10);
    assert(x == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  test_default_template_arg();
  static_assert(test_default_template_arg());

  return 0;
}
