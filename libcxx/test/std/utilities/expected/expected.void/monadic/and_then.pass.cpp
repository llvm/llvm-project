//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <expected>

// template<class F> constexpr auto and_then(F&& f) &;
// template<class F> constexpr auto and_then(F&& f) const &;
// template<class F> constexpr auto and_then(F&& f) &&;
// template<class F> constexpr auto and_then(F&& f) const &&;

#include <expected>
#include <concepts>
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

struct NonCopyable {
  constexpr NonCopyable(int) {}
  NonCopyable(const NonCopyable&) = delete;
};

struct NonMovable {
  constexpr NonMovable(int) {}
  NonMovable(NonMovable&&) = delete;
};

template <class E, class F>
concept has_and_then =
    requires(E&& e, F&& f) {
      { std::forward<E>(e).and_then(std::forward<F>(f)) };
    };

std::expected<void, int> return_int() { return {}; }
std::expected<void, NonCopyable> return_noncopyable() { return {}; }
std::expected<void, NonMovable> return_nonmovable() { return {}; }

static_assert(has_and_then<std::expected<void, int>&, decltype(return_int)>);
static_assert(!has_and_then<std::expected<void, NonCopyable>&, decltype(return_noncopyable)>);
static_assert(has_and_then<const std::expected<void, int>&, decltype(return_int)>);
static_assert(!has_and_then<const std::expected<void, NonCopyable>&, decltype(return_noncopyable)>);
static_assert(has_and_then<std::expected<void, int>&&, decltype(return_int)>);
static_assert(!has_and_then<std::expected<void, NonMovable>&&, decltype(return_nonmovable)>);
static_assert(has_and_then<const std::expected<void, int>&&, decltype(return_int)>);
static_assert(!has_and_then<const std::expected<void, NonMovable>&&, decltype(return_nonmovable)>);

// [LWG 3877] https://cplusplus.github.io/LWG/issue3877, check constraint failing but not compile error inside the function body.
static_assert(!has_and_then<const std::expected<int, std::unique_ptr<int>>&, int()>);
static_assert(!has_and_then<const std::expected<int, std::unique_ptr<int>>&&, int()>);

constexpr void test_val_types() {
  // Test & overload
  {
    auto l = [] -> std::expected<int, int> { return 2; };
    std::expected<void, int> v;
    std::same_as<std::expected<int, int>> decltype(auto) val = v.and_then(l);
    assert(val == 2);
  }

  // Test const& overload
  {
    auto l = [] -> std::expected<int, int> { return 2; };
    const std::expected<void, int> v;
    assert(v.and_then(l).value() == 2);
    static_assert(std::is_same_v< decltype(v.and_then(l)), std::expected<int, int>>);
  }

  // Test && overload
  {
    auto l = [] -> std::expected<int, int> { return 2; };
    std::expected<void, int> v;
    std::same_as<std::expected<int, int>> decltype(auto) val = std::move(v).and_then(l);
    assert(val == 2);
  }

  // Test const&& overload
  {
    auto l = [] -> std::expected<int, int> { return 2; };
    const std::expected<void, int> v;
    std::same_as<std::expected<int, int>> decltype(auto) val = std::move(v).and_then(l);
    assert(val == 2);
  }
}

constexpr void test_fail() {
  // Test & overload
  {
    auto f = [] -> std::expected<int, int> {
      assert(false);
      return 0;
    };
    std::expected<void, int> v(std::unexpected<int>(2));
    std::same_as<std::expected<int, int>> decltype(auto) val = v.and_then(f);
    assert(val.error() == 2);
  }

  // Test const& overload
  {
    auto f = [] -> std::expected<int, int> {
      assert(false);
      return 0;
    };
    const std::expected<void, int> v(std::unexpected<int>(2));
    std::same_as<std::expected<int, int>> decltype(auto) val = v.and_then(f);
    assert(val.error() == 2);
  }

  // Test && overload
  {
    auto f = [] -> std::expected<int, int> {
      assert(false);
      return 0;
    };
    std::expected<void, int> v(std::unexpected<int>(2));
    std::same_as<std::expected<int, int>> decltype(auto) val = std::move(v).and_then(f);
    assert(val.error() == 2);
  }

  // Test const&& overload
  {
    auto f = [] -> std::expected<int, int> {
      assert(false);
      return 0;
    };
    const std::expected<void, int> v(std::unexpected<int>(2));
    std::same_as<std::expected<int, int>> decltype(auto) val = std::move(v).and_then(f);
    assert(val.error() == 2);
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
