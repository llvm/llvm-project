//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <expected>

// template<class F> constexpr auto transform_error(F&& f) &;
// template<class F> constexpr auto transform_error(F&& f) const &;
// template<class F> constexpr auto transform_error(F&& f) &&;
// template<class F> constexpr auto transform_error(F&& f) const &&;

#include <expected>
#include <concepts>
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

template <class E, class F>
concept has_transform =
    requires(E&& e, F&& f) {
      { std::forward<E>(e).transform(std::forward<F>(f)) };
    };

// [LWG 3877] https://cplusplus.github.io/LWG/issue3877, check constraint failing but not compile error inside the function body.
static_assert(!has_transform<const std::expected<int, std::unique_ptr<int>>&, int()>);
static_assert(!has_transform<const std::expected<int, std::unique_ptr<int>>&&, int()>);

constexpr void test_val_types() {
  // Test & overload
  {
    auto l = [] -> int { return 1; };
    std::expected<void, int> v;
    std::same_as<std::expected<int, int>> decltype(auto) val = v.transform(l);
    assert(val == 1);
  }

  // Test const& overload
  {
    auto l = [] -> int { return 1; };
    const std::expected<void, int> v;
    std::same_as<std::expected<int, int>> decltype(auto) val = v.transform(l);
    assert(val == 1);
  }

  // Test && overload
  {
    auto l = [] -> int { return 1; };
    std::expected<void, int> v;
    std::same_as<std::expected<int, int>> decltype(auto) val = std::move(v).transform(l);
    assert(val == 1);
  }

  // Test const&& overload
  {
    auto l = [] -> int { return 1; };
    const std::expected<void, int> v;
    std::same_as<std::expected<int, int>> decltype(auto) val = std::move(v).transform(l);
    assert(val == 1);
  }
}

constexpr void test_fail() {
  // Test & overload
  {
    auto l = [] -> int {
      assert(false);
      return 0;
    };
    std::expected<void, int> v(std::unexpected<int>(5));
    std::same_as<std::expected<int, int>> decltype(auto) val = v.transform(l);
    assert(val.error() == 5);
  }

  // Test const& overload
  {
    auto l = [] -> int {
      assert(false);
      return 0;
    };
    const std::expected<void, int> v(std::unexpected<int>(5));
    std::same_as<std::expected<int, int>> decltype(auto) val = v.transform(l);
    assert(val.error() == 5);
  }

  // Test && overload
  {
    auto l = [] -> int {
      assert(false);
      return 0;
    };
    std::expected<void, int> v(std::unexpected<int>(5));
    std::same_as<std::expected<int, int>> decltype(auto) val = std::move(v).transform(l);
    assert(val.error() == 5);
  }

  // Test const&& overload
  {
    auto l = [] -> int {
      assert(false);
      return 0;
    };
    const std::expected<void, int> v(std::unexpected<int>(5));
    std::same_as<std::expected<int, int>> decltype(auto) val = std::move(v).transform(l);
    assert(val.error() == 5);
  }
}

constexpr bool test() {
  test_fail();
  test_val_types();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
