//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// GCC has a issue for `Guaranteed copy elision for potentially-overlapping non-static data members`,
// please refer to: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108333, but we have a workaround to
// avoid this issue.

// <expected>

// template<class F> constexpr auto transform(F&& f) &;
// template<class F> constexpr auto transform(F&& f) const &;
// template<class F> constexpr auto transform(F&& f) &&;
// template<class F> constexpr auto transform(F&& f) const &&;

#include <expected>
#include <concepts>
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

#include "../../types.h"

struct LVal {
  constexpr int operator()(int&) { return 1; }
  int operator()(const int&)  = delete;
  int operator()(int&&)       = delete;
  int operator()(const int&&) = delete;
};

struct CLVal {
  int operator()(int&) = delete;
  constexpr int operator()(const int&) { return 1; }
  int operator()(int&&)       = delete;
  int operator()(const int&&) = delete;
};

struct RVal {
  int operator()(int&)       = delete;
  int operator()(const int&) = delete;
  constexpr int operator()(int&&) { return 1; }
  int operator()(const int&&) = delete;
};

struct CRVal {
  int operator()(int&)       = delete;
  int operator()(const int&) = delete;
  int operator()(int&&)      = delete;
  constexpr int operator()(const int&&) { return 1; }
};

struct RefQual {
  constexpr int operator()(int) & { return 1; }
  int operator()(int) const&  = delete;
  int operator()(int) &&      = delete;
  int operator()(int) const&& = delete;
};

struct CRefQual {
  int operator()(int) & = delete;
  constexpr int operator()(int) const& { return 1; }
  int operator()(int) &&      = delete;
  int operator()(int) const&& = delete;
};

struct RVRefQual {
  int operator()(int) &      = delete;
  int operator()(int) const& = delete;
  constexpr int operator()(int) && { return 1; }
  int operator()(int) const&& = delete;
};

struct RVCRefQual {
  int operator()(int) &      = delete;
  int operator()(int) const& = delete;
  int operator()(int) &&     = delete;
  constexpr int operator()(int) const&& { return 1; }
};

struct NonCopy {
  int value;
  constexpr explicit NonCopy(int val) : value(val) {}
  NonCopy(const NonCopy&) = delete;
};

struct NonConst {
  int non_const() { return 1; }
};

template <class E, class F>
concept has_transform =
    requires(E&& e, F&& f) {
      { std::forward<E>(e).transform(std::forward<F>(f)) };
    };

// clang-format off
// [LWG 3877] https://cplusplus.github.io/LWG/issue3877, check constraint failing but not compile error inside the function body.
static_assert(!has_transform<const std::expected<int, std::unique_ptr<int>>&, int()>);
static_assert(!has_transform<const std::expected<int, std::unique_ptr<int>>&&, int()>);

// [LWG 3983] https://cplusplus.github.io/LWG/issue3938, check std::expected monadic ops well-formed with move-only error_type.
// There are no effects for `&` and `const &` overload, because the constraints requires is_constructible_v<E, decltype(error())> is true.
static_assert(has_transform<std::expected<int, MoveOnlyErrorType>&&, int(int)>);
static_assert(has_transform<const std::expected<int, MoveOnlyErrorType>&&, int(const int)>);

constexpr void test_val_types() {
  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      std::expected<int, int> e(0);
      std::same_as<std::expected<int, int>> decltype(auto) val = e.transform(LVal{});
      assert(val == 1);
    }

    // With & qualifier on F's operator()
    {
      std::expected<int, int> e(0);
      RefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.transform(l);
      assert(val == 1);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const std::expected<int, int> e(0);
      std::same_as<std::expected<int, int>> decltype(auto) val = e.transform(CLVal{});
      assert(val == 1);
    }

    // With & qualifier on F's operator()
    {
      const std::expected<int, int> e(0);
      const CRefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.transform(l);
      assert(val == 1);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      std::expected<int, int> e(0);
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).transform(RVal{});
      assert(val == 1);
    }

    // With & qualifier on F's operator()
    {
      std::expected<int, int> e(0);
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).transform(RVRefQual{});
      assert(val == 1);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const std::expected<int, int> e(0);
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).transform(CRVal{});
      assert(val == 1);
    }

    // With & qualifier on F's operator()
    {
      const std::expected<int, int> e(0);
      const RVCRefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.transform(std::move(l));
      assert(val == 1);
    }
  }
}
// clang-format on

constexpr void test_take_val_return_void() {
  std::expected<int, int> e(1);
  int val = 0;
  e.transform([&val]<typename T>(T&&) -> void {
    static_assert(std::is_same_v<T, int&>);
    assert(val == 0);
    val = 1;
  });
  assert(val == 1);
  std::move(e).transform([&val]<typename T>(T&&) -> void {
    static_assert(std::is_same_v<T, int>);
    assert(val == 1);
    val = 2;
  });

  const auto& ce = e;
  assert(val == 2);
  ce.transform([&val]<typename T>(T&&) -> void {
    static_assert(std::is_same_v<T, const int&>);
    assert(val == 2);
    val = 3;
  });
  assert(val == 3);
  std::move(ce).transform([&val]<typename T>(T&&) -> void {
    static_assert(std::is_same_v<T, const int>);
    assert(val == 3);
    val = 4;
  });
  assert(val == 4);
}

// check val member is direct-non-list-initialized with invoke(std::forward<F>(f), value())
constexpr void test_direct_non_list_init() {
  auto xform = [](int i) { return NonCopy(i); };
  std::expected<int, int> e(2);
  std::expected<NonCopy, int> n = e.transform(xform);
  assert(n.value().value == 2);
}

// check that the lambda body is not instantiated during overload resolution
constexpr void test_sfinae() {
  std::expected<NonConst, int> e(std::unexpected<int>(2));
  auto l = [](auto&& x) { return x.non_const(); };
  e.transform(l);
  std::move(e).transform(l);

  std::expected<int, int> e1(std::unexpected<int>(1));
  const auto& ce1         = e1;
  const auto never_called = [](int) {
    assert(false);
    return std::expected<int, int>();
  };

  e1.transform(never_called);
  std::move(e1).transform(never_called);
  ce1.and_then(never_called);
  std::move(ce1).transform(never_called);
}

constexpr void test_move_only_error_type() {
  // Test &&
  {
      std::expected<int, MoveOnlyErrorType> e;
      auto l = [](int) { return 0; };
      std::move(e).transform(l);
  }

  // Test const&&
  {
      const std::expected<int, MoveOnlyErrorType> e;
      auto l = [](const int) { return 0; };
      std::move(e).transform(l);
  }
}

constexpr bool test() {
  test_sfinae();
  test_val_types();
  test_direct_non_list_init();
  test_move_only_error_type();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
