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

struct UnexpectedLVal {
  constexpr std::expected<int, int> operator()(int&) { return std::expected<int, int>(std::unexpected<int>(5)); }
  std::expected<int, int> operator()(const int&)  = delete;
  std::expected<int, int> operator()(int&&)       = delete;
  std::expected<int, int> operator()(const int&&) = delete;
};

struct UnexpectedCLVal {
  std::expected<int, int> operator()(int&) = delete;
  constexpr std::expected<int, int> operator()(const int&) { return std::expected<int, int>(std::unexpected<int>(5)); }
  std::expected<int, int> operator()(int&&)       = delete;
  std::expected<int, int> operator()(const int&&) = delete;
};

struct UnexpectedRVal {
  std::expected<int, int> operator()(int&)       = delete;
  std::expected<int, int> operator()(const int&) = delete;
  constexpr std::expected<int, int> operator()(int&&) { return std::expected<int, int>(std::unexpected<int>(5)); }
  std::expected<int, int> operator()(const int&&) = delete;
};

struct UnexpectedCRVal {
  std::expected<int, int> operator()(int&)       = delete;
  std::expected<int, int> operator()(const int&) = delete;
  std::expected<int, int> operator()(int&&)      = delete;
  constexpr std::expected<int, int> operator()(const int&&) { return std::expected<int, int>(std::unexpected<int>(5)); }
};

struct UnexpectedRefQual {
  constexpr std::expected<int, int> operator()(int) & { return std::expected<int, int>(std::unexpected<int>(5)); }
  std::expected<int, int> operator()(int) const&  = delete;
  std::expected<int, int> operator()(int) &&      = delete;
  std::expected<int, int> operator()(int) const&& = delete;
};

struct UnexpectedCRefQual {
  std::expected<int, int> operator()(int) & = delete;
  constexpr std::expected<int, int> operator()(int) const& { return std::expected<int, int>(std::unexpected<int>(5)); }
  std::expected<int, int> operator()(int) &&      = delete;
  std::expected<int, int> operator()(int) const&& = delete;
};

struct UnexpectedRVRefQual {
  std::expected<int, int> operator()(int) &      = delete;
  std::expected<int, int> operator()(int) const& = delete;
  constexpr std::expected<int, int> operator()(int) && { return std::expected<int, int>(std::unexpected<int>(5)); }
  std::expected<int, int> operator()(int) const&& = delete;
};

struct UnexpectedRVCRefQual {
  std::expected<int, int> operator()(int) &      = delete;
  std::expected<int, int> operator()(int) const& = delete;
  std::expected<int, int> operator()(int) &&     = delete;
  constexpr std::expected<int, int> operator()(int) const&& { return std::expected<int, int>(std::unexpected<int>(5)); }
};

struct NonCopyable {
  constexpr NonCopyable(int) {}
  NonCopyable(const NonCopyable&) = delete;
};

struct NonMovable {
  constexpr NonMovable(int) {}
  NonMovable(NonMovable&&) = delete;
};

struct NonConst {
  std::expected<int, int> non_const() { return 1; }
};

template <class E, class F>
concept has_and_then = requires(E&& e, F&& f) {
  {std::forward<E>(e).and_then(std::forward<F>(f))};
};

// clang-format off
static_assert( has_and_then<std::expected<int, int>&, std::expected<int, int>(int&)>);
static_assert(!has_and_then<std::expected<int, NonCopyable>&, std::expected<int, NonCopyable>(int&)>);
static_assert( has_and_then<const std::expected<int, int>&, std::expected<int, int>(const int&)>);
static_assert(!has_and_then<const std::expected<int, NonCopyable>&, std::expected<int, NonCopyable>(const int&)>);
static_assert( has_and_then<std::expected<int, int>&&, std::expected<int, int>(int)>);
static_assert(!has_and_then<std::expected<int, NonMovable>&&, std::expected<int, NonMovable>(int)>);
static_assert( has_and_then<const std::expected<int, int>&&, std::expected<int, int>(const int)>);
static_assert(!has_and_then<const std::expected<int, NonMovable>&&, std::expected<int, NonMovable>(const int)>);

// [LWG 3877] https://cplusplus.github.io/LWG/issue3877, check constraint failing but not compile error inside the function body.
static_assert(!has_and_then<const std::expected<int, std::unique_ptr<int>>&, int()>);
static_assert(!has_and_then<const std::expected<int, std::unique_ptr<int>>&&, int()>);

// [LWG 3983] https://cplusplus.github.io/LWG/issue3938, check std::expected monadic ops well-formed with move-only error_type.
// There are no effects for `&` and `const &` overload, because the constraints requires is_constructible_v<E, decltype(error())> is true.
static_assert(has_and_then<std::expected<int, MoveOnlyErrorType>&&, std::expected<int, MoveOnlyErrorType>(int)>);
static_assert(has_and_then<const std::expected<int, MoveOnlyErrorType>&&, std::expected<int, MoveOnlyErrorType>(const int)>);

constexpr void test_val_types() {
  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      std::expected<int, int> e{0};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.and_then(LVal{});
      assert(val == 1);
      assert(e.and_then(UnexpectedLVal{}).error() == 5);
    }

    // With & qualifier on F's operator()
    {
      std::expected<int, int> e{0};
      RefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.and_then(l);
      assert(val == 1);
      UnexpectedRefQual nl{};
      assert(e.and_then(nl).error() == 5);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const std::expected<int, int> e{0};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.and_then(CLVal{});
      assert(val == 1);
      assert(e.and_then(UnexpectedCLVal{}).error() == 5);
    }

    // With & qualifier on F's operator()
    {
      const std::expected<int, int> e{0};
      const CRefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.and_then(l);
      assert(val == 1);
      const UnexpectedCRefQual nl{};
      assert(e.and_then(nl).error() == 5);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      std::expected<int, int> e{0};
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).and_then(RVal{});
      assert(val == 1);
      assert(std::move(e).and_then(UnexpectedRVal{}).error() == 5);
    }

    // With & qualifier on F's operator()
    {
      std::expected<int, int> e{0};
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).and_then(RVRefQual{});
      assert(val == 1);
      assert(e.and_then(UnexpectedRVRefQual{}).error() == 5);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const std::expected<int, int> e{0};
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).and_then(CRVal{});
      assert(val == 1);
      assert(std::move(e).and_then(UnexpectedCRVal{}).error() == 5);
    }

    // With & qualifier on F's operator()
    {
      const std::expected<int, int> e{0};
      const RVCRefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).and_then(std::move(l));
      assert(val == 1);
      const UnexpectedRVCRefQual nl{};
      assert(std::move(e).and_then(std::move(nl)).error() == 5);
    }
  }
}
// clang-format on

// check that the lambda body is not instantiated during overload resolution
constexpr void test_sfinae() {
  std::expected<NonConst, int> e(std::unexpected<int>(2));
  auto l = [](auto&& x) { return x.non_const(); };
  e.and_then(l);
  std::move(e).and_then(l);
}

constexpr void test_move_only_error_type() {
  // Test &&
  {
    std::expected<int, MoveOnlyErrorType> e;
    auto l = [](int) { return std::expected<int, MoveOnlyErrorType>{}; };
    std::move(e).and_then(l);
  }

  // Test const&&
  {
    const std::expected<int, MoveOnlyErrorType> e;
    auto l = [](const int) { return std::expected<int, MoveOnlyErrorType>{}; };
    std::move(e).and_then(l);
  }
}

constexpr bool test() {
  test_sfinae();
  test_val_types();
  test_move_only_error_type();

  std::expected<int, int> e(std::unexpected<int>(1));
  const auto& ce = e;

  const auto never_called = [](int) {
    assert(false);
    return std::expected<int, int>();
  };

  e.and_then(never_called);
  std::move(e).and_then(never_called);
  ce.and_then(never_called);
  std::move(ce).and_then(never_called);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
