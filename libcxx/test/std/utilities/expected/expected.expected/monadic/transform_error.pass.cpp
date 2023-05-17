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
concept has_transform_error =
    requires(E&& e, F&& f) {
      { std::forward<E>(e).transform_error(std::forward<F>(f)) };
    };

// [LWG 3877] https://cplusplus.github.io/LWG/issue3877, check constraint failing but not compile error inside the function body.
static_assert(!has_transform_error<const std::expected<std::unique_ptr<int>, int>&, int()>);
static_assert(!has_transform_error<const std::expected<std::unique_ptr<int>, int>&&, int()>);

// clang-format off
constexpr void test_val_types() {
  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = e.transform_error(LVal{});
      assert(val.error() == 1);
    }

    // With & qualifier on F's operator()
    {
      std::expected<int, int> e(std::unexpected<int>(0));
      RefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.transform_error(l);
      assert(val.error() == 1);
    }
  }

  // Test const& overload
  {
    // Without const& qualifier on F's operator()
    {
      const std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = e.transform_error(CLVal{});
      assert(val.error() == 1);
    }

    // With const& qualifier on F's operator()
    {
      const std::expected<int, int> e(std::unexpected<int>(0));
      const CRefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = e.transform_error(l);
      assert(val.error() == 1);
    }
  }

  // Test && overload
  {
    // Without && qualifier on F's operator()
    {
      std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).transform_error(RVal{});
      assert(val.error() == 1);
    }

    // With && qualifier on F's operator()
    {
      std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).transform_error(RVRefQual{});
      assert(val.error() == 1);
    }
  }

  // Test const&& overload
  {
    // Without const&& qualifier on F's operator()
    {
      const std::expected<int, int> e(std::unexpected<int>(0));
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).transform_error(CRVal{});
      assert(val.error() == 1);
    }

    // With const&& qualifier on F's operator()
    {
      const std::expected<int, int> e(std::unexpected<int>(0));
      const RVCRefQual l{};
      std::same_as<std::expected<int, int>> decltype(auto) val = std::move(e).transform_error(std::move(l));
      assert(val.error() == 1);
    }
  }
}
// clang-format on

// check unex member is direct-non-list-initialized with invoke(std::forward<F>(f), error())
constexpr void test_direct_non_list_init() {
  auto xform = [](int i) { return NonCopy(i); };
  std::expected<int, int> e(std::unexpected<int>(2));
  std::expected<int, NonCopy> n = e.transform_error(xform);
  assert(n.error().value == 2);
}

// check that the lambda body is not instantiated during overload resolution
constexpr void test_sfinae() {
  std::expected<int, NonConst> e(2);
  auto l = [](auto&& x) { return x.non_const(); };
  e.transform_error(l);
  std::move(e).transform_error(l);

  std::expected<int, int> e1;
  const auto& ce1 = e1;

  const auto never_called = [](int) {
    assert(false);
    return 0;
  };

  e1.transform_error(never_called);
  std::move(e1).transform_error(never_called);
  ce1.transform_error(never_called);
  std::move(ce1).transform_error(never_called);
}

constexpr bool test() {
  test_sfinae();
  test_val_types();
  test_direct_non_list_init();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
