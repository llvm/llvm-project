//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <expected>

// template<class F> constexpr auto or_else(F&& f) &;
// template<class F> constexpr auto or_else(F&& f) const &;
// template<class F> constexpr auto or_else(F&& f) &&;
// template<class F> constexpr auto or_else(F&& f) const &&;

#include <cassert>
#include <concepts>
#include <expected>
#include <memory>
#include <type_traits>
#include <utility>

#include "../../types.h"

struct LVal {
  constexpr std::expected<int, int> operator()(int&) { return 1; }
  std::expected<int, int> operator()(const int&)  = delete;
  std::expected<int, int> operator()(int&&)       = delete;
  std::expected<int, int> operator()(const int&&) = delete;
};

struct CLVal {
  std::expected<int, int> operator()(int&) = delete;
  constexpr std::expected<int, int> operator()(const int&) { return 1; }
  std::expected<int, int> operator()(int&&)       = delete;
  std::expected<int, int> operator()(const int&&) = delete;
};

struct RVal {
  std::expected<int, int> operator()(int&)       = delete;
  std::expected<int, int> operator()(const int&) = delete;
  constexpr std::expected<int, int> operator()(int&&) { return 1; }
  std::expected<int, int> operator()(const int&&) = delete;
};

struct CRVal {
  std::expected<int, int> operator()(int&)       = delete;
  std::expected<int, int> operator()(const int&) = delete;
  std::expected<int, int> operator()(int&&)      = delete;
  constexpr std::expected<int, int> operator()(const int&&) { return 1; }
};

struct RefQual {
  constexpr std::expected<int, int> operator()(int) & { return 1; }
  std::expected<int, int> operator()(int) const&  = delete;
  std::expected<int, int> operator()(int) &&      = delete;
  std::expected<int, int> operator()(int) const&& = delete;
};

struct CRefQual {
  std::expected<int, int> operator()(int) & = delete;
  constexpr std::expected<int, int> operator()(int) const& { return 1; }
  std::expected<int, int> operator()(int) &&      = delete;
  std::expected<int, int> operator()(int) const&& = delete;
};

struct RVRefQual {
  std::expected<int, int> operator()(int) &      = delete;
  std::expected<int, int> operator()(int) const& = delete;
  constexpr std::expected<int, int> operator()(int) && { return 1; }
  std::expected<int, int> operator()(int) const&& = delete;
};

struct RVCRefQual {
  std::expected<int, int> operator()(int) &      = delete;
  std::expected<int, int> operator()(int) const& = delete;
  std::expected<int, int> operator()(int) &&     = delete;
  constexpr std::expected<int, int> operator()(int) const&& { return 1; }
};

template <class E, class F>
concept has_or_else =
    requires(E&& e, F&& f) {
      { std::forward<E>(e).or_else(std::forward<F>(f)) };
    };
// clang-format off
// [LWG 3877] https://cplusplus.github.io/LWG/issue3877, check constraint failing but not compile error inside the function body.
static_assert(!has_or_else<const std::expected<std::unique_ptr<int>, int>&, int()>);
static_assert(!has_or_else<const std::expected<std::unique_ptr<int>, int>&&, int()>);

// [LWG 3983] https://cplusplus.github.io/LWG/issue3938, check std::expected monadic ops well-formed with move-only error_type.
static_assert(has_or_else<std::expected<int, MoveOnlyErrorType>&, std::expected<int, int>(MoveOnlyErrorType &)>);
static_assert(has_or_else<const std::expected<int, MoveOnlyErrorType>&, std::expected<int, int>(const MoveOnlyErrorType &)>);
static_assert(has_or_else<std::expected<int, MoveOnlyErrorType>&&, std::expected<int, int>(MoveOnlyErrorType&&)>);
static_assert(has_or_else<const std::expected<int, MoveOnlyErrorType>&&, std::expected<int, int>(const MoveOnlyErrorType&&)>);

constexpr void test_val_types() {
  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = e.or_else(LVal{});
      assert(val == 1);
    }

    // With & qualifier on F's operator
    {
      std::expected<int, int> e(std::unexpected<int>(0));
      RefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.or_else(l);
      assert(val == 1);
    }
  }

  // Test const& overload
  {
    // Without const& qualifier on F's operator()
    {
      const std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = e.or_else(CLVal{});
      assert(val == 1);
    }

    // With const& qualifier on F's operator()
    {
      const std::expected<int, int> e(std::unexpected<int>(0));
      const CRefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.or_else(l);
      assert(val == 1);
    }
  }

  // Test && overload
  {
    // Without && qualifier on F's operator()
    {
      std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).or_else(RVal{});
      assert(val == 1);
    }

    // With && qualifier on F's operator()
    {
      std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).or_else(RVRefQual{});
      assert(val == 1);
    }
  }

  // Test const&& overload
  {
    // Without const&& qualifier on F's operator()
    {
      const std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).or_else(CRVal{});
      assert(val == 1);
    }

    // With const&& qualifier on F's operator()
    {
      const std::expected<int, int> e(std::unexpected<int>(0));
      const RVCRefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).or_else(std::move(l));
      assert(val == 1);
    }
  }
}
// clang-format on

struct NonConst {
  std::expected<int, int> non_const() { return std::expected<int, int>(std::unexpect, 1); }
};

// check that the lambda body is not instantiated during overload resolution
constexpr void test_sfinae() {
  std::expected<int, NonConst> e{1};
  auto l = [](auto&& x) { return x.non_const(); };
  (void)e.or_else(l);
  (void)std::move(e).or_else(l);
}

constexpr void test_move_only_error_type() {
  // Test &
  {
      std::expected<int, MoveOnlyErrorType> e;
      auto l = [](MoveOnlyErrorType&) { return std::expected<int, int>{}; };
      (void)e.or_else(l);
  }

  // Test const&
  {
      const std::expected<int, MoveOnlyErrorType> e;
      auto l = [](const MoveOnlyErrorType&) { return std::expected<int, int>{}; };
      (void)e.or_else(l);
  }

  // Test &&
  {
      std::expected<int, MoveOnlyErrorType> e;
      auto l = [](MoveOnlyErrorType&&) { return std::expected<int, int>{}; };
      (void)std::move(e).or_else(l);
  }

  // Test const&&
  {
      const std::expected<int, MoveOnlyErrorType> e;
      auto l = [](const MoveOnlyErrorType&&) { return std::expected<int, int>{}; };
      (void)std::move(e).or_else(l);
  }
}

constexpr bool test() {
  test_sfinae();
  test_val_types();
  test_move_only_error_type();

  std::expected<int, int> e(1);
  const auto& ce = e;

  const auto never_called = [](int) {
    assert(false);
    return std::expected<int, int>();
  };

  (void)e.or_else(never_called);
  (void)std::move(e).or_else(never_called);
  (void)ce.or_else(never_called);
  (void)std::move(ce).or_else(never_called);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
