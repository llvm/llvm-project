//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <optional>

// template<class F> constexpr auto and_then(F&&) &;
// template<class F> constexpr auto and_then(F&&) &&;
// template<class F> constexpr auto and_then(F&&) const&;
// template<class F> constexpr auto and_then(F&&) const&&;

#include <cassert>
#include <concepts>
#include <optional>
#include <utility>

#include "test_macros.h"

struct LVal {
  constexpr std::optional<int> operator()(int&) { return 1; }
  std::optional<int> operator()(const int&) = delete;
  std::optional<int> operator()(int&&) = delete;
  std::optional<int> operator()(const int&&) = delete;
};

struct CLVal {
  std::optional<int> operator()(int&) = delete;
  constexpr std::optional<int> operator()(const int&) { return 1; }
  std::optional<int> operator()(int&&) = delete;
  std::optional<int> operator()(const int&&) = delete;
};

struct RVal {
  std::optional<int> operator()(int&) = delete;
  std::optional<int> operator()(const int&) = delete;
  constexpr std::optional<int> operator()(int&&) { return 1; }
  std::optional<int> operator()(const int&&) = delete;
};

struct CRVal {
  std::optional<int> operator()(int&) = delete;
  std::optional<int> operator()(const int&) = delete;
  std::optional<int> operator()(int&&) = delete;
  constexpr std::optional<int> operator()(const int&&) { return 1; }
};

struct RefQual {
  constexpr std::optional<int> operator()(int) & { return 1; }
  std::optional<int> operator()(int) const& = delete;
  std::optional<int> operator()(int) && = delete;
  std::optional<int> operator()(int) const&& = delete;
};

struct CRefQual {
  std::optional<int> operator()(int) & = delete;
  constexpr std::optional<int> operator()(int) const& { return 1; }
  std::optional<int> operator()(int) && = delete;
  std::optional<int> operator()(int) const&& = delete;
};

struct RVRefQual {
  std::optional<int> operator()(int) & = delete;
  std::optional<int> operator()(int) const& = delete;
  constexpr std::optional<int> operator()(int) && { return 1; }
  std::optional<int> operator()(int) const&& = delete;
};

struct RVCRefQual {
  std::optional<int> operator()(int) & = delete;
  std::optional<int> operator()(int) const& = delete;
  std::optional<int> operator()(int) && = delete;
  constexpr std::optional<int> operator()(int) const&& { return 1; }
};

struct NOLVal {
  constexpr std::optional<int> operator()(int&) { return std::nullopt; }
  std::optional<int> operator()(const int&) = delete;
  std::optional<int> operator()(int&&) = delete;
  std::optional<int> operator()(const int&&) = delete;
};

struct NOCLVal {
  std::optional<int> operator()(int&) = delete;
  constexpr std::optional<int> operator()(const int&) { return std::nullopt; }
  std::optional<int> operator()(int&&) = delete;
  std::optional<int> operator()(const int&&) = delete;
};

struct NORVal {
  std::optional<int> operator()(int&) = delete;
  std::optional<int> operator()(const int&) = delete;
  constexpr std::optional<int> operator()(int&&) { return std::nullopt; }
  std::optional<int> operator()(const int&&) = delete;
};

struct NOCRVal {
  std::optional<int> operator()(int&) = delete;
  std::optional<int> operator()(const int&) = delete;
  std::optional<int> operator()(int&&) = delete;
  constexpr std::optional<int> operator()(const int&&) { return std::nullopt; }
};

struct NORefQual {
  constexpr std::optional<int> operator()(int) & { return std::nullopt; }
  std::optional<int> operator()(int) const& = delete;
  std::optional<int> operator()(int) && = delete;
  std::optional<int> operator()(int) const&& = delete;
};

struct NOCRefQual {
  std::optional<int> operator()(int) & = delete;
  constexpr std::optional<int> operator()(int) const& { return std::nullopt; }
  std::optional<int> operator()(int) && = delete;
  std::optional<int> operator()(int) const&& = delete;
};

struct NORVRefQual {
  std::optional<int> operator()(int) & = delete;
  std::optional<int> operator()(int) const& = delete;
  constexpr std::optional<int> operator()(int) && { return std::nullopt; }
  std::optional<int> operator()(int) const&& = delete;
};

struct NORVCRefQual {
  std::optional<int> operator()(int) & = delete;
  std::optional<int> operator()(int) const& = delete;
  std::optional<int> operator()(int) && = delete;
  constexpr std::optional<int> operator()(int) const&& { return std::nullopt; }
};

struct NoCopy {
  NoCopy() = default;
  NoCopy(const NoCopy&) { assert(false); }
  std::optional<int> operator()(const NoCopy&&) { return 1; }
};

struct NonConst {
  std::optional<int> non_const() { return 1; }
};

constexpr void test_val_types() {
  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      std::optional<int> i{0};
      assert(i.and_then(LVal{}) == 1);
      assert(i.and_then(NOLVal{}) == std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(LVal{})), std::optional<int>);
    }

    //With & qualifier on F's operator()
    {
      std::optional<int> i{0};
      RefQual l{};
      assert(i.and_then(l) == 1);
      NORefQual nl{};
      assert(i.and_then(nl) == std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), std::optional<int>);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const std::optional<int> i{0};
      assert(i.and_then(CLVal{}) == 1);
      assert(i.and_then(NOCLVal{}) == std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(CLVal{})), std::optional<int>);
    }

    //With & qualifier on F's operator()
    {
      const std::optional<int> i{0};
      const CRefQual l{};
      assert(i.and_then(l) == 1);
      const NOCRefQual nl{};
      assert(i.and_then(nl) == std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), std::optional<int>);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      std::optional<int> i{0};
      assert(std::move(i).and_then(RVal{}) == 1);
      assert(std::move(i).and_then(NORVal{}) == std::nullopt);
      ASSERT_SAME_TYPE(decltype(std::move(i).and_then(RVal{})), std::optional<int>);
    }

    //With & qualifier on F's operator()
    {
      std::optional<int> i{0};
      assert(i.and_then(RVRefQual{}) == 1);
      assert(i.and_then(NORVRefQual{}) == std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(RVRefQual{})), std::optional<int>);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const std::optional<int> i{0};
      assert(std::move(i).and_then(CRVal{}) == 1);
      assert(std::move(i).and_then(NOCRVal{}) == std::nullopt);
      ASSERT_SAME_TYPE(decltype(std::move(i).and_then(CRVal{})), std::optional<int>);
    }

    //With & qualifier on F's operator()
    {
      const std::optional<int> i{0};
      const RVCRefQual l{};
      assert(i.and_then(std::move(l)) == 1);
      const NORVCRefQual nl{};
      assert(i.and_then(std::move(nl)) == std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(std::move(l))), std::optional<int>);
    }
  }
}

// check that the lambda body is not instantiated during overload resolution
constexpr void test_sfinae() {
  std::optional<NonConst> opt{};
  auto l = [](auto&& x) { return x.non_const(); };
  (void)opt.and_then(l);
  (void)std::move(opt).and_then(l);
}

constexpr bool test() {
  test_val_types();
  std::optional<int> opt{};
  const auto& copt = opt;

  const auto never_called = [](int) {
    assert(false);
    return std::optional<int>{};
  };

  (void)opt.and_then(never_called);
  (void)std::move(opt).and_then(never_called);
  (void)copt.and_then(never_called);
  (void)std::move(copt).and_then(never_called);

  std::optional<NoCopy> nc;
  const auto& cnc = nc;
  (void)std::move(cnc).and_then(NoCopy{});
  (void)std::move(nc).and_then(NoCopy{});

  return true;
}

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  // Test "overloads", only the const (no ref-qualifier) and_then() should be called

  { // &
    // Without & qualifier on F's operator()
    {
      int j = 42;
      std::optional<int&> i{j};
      std::same_as<std::optional<int>> decltype(auto) r = i.and_then(LVal{});

      assert(r == 1);
      assert(i.and_then(NOLVal{}) == std::nullopt);
    }

    //With & qualifier on F's operator()
    {
      int j = 42;
      std::optional<int&> i{j};
      auto& io = i;
      RefQual l{};
      NORefQual nl{};
      std::same_as<std::optional<int>> decltype(auto) r = io.and_then(l);

      assert(r == 1);
      assert(io.and_then(nl) == std::nullopt);
    }
  }

  { // const&
    // Without & qualifier on F's operator()
    {
      int j = 42;
      const std::optional<const int&> i{j};
      std::same_as<std::optional<int>> decltype(auto) r = i.and_then(CLVal{});

      assert(r == 1);
      assert(i.and_then(NOCLVal{}) == std::nullopt);
    }

    //With & qualifier on F's operator()
    {
      int j = 42;
      const std::optional<int&> i{j};
      const CRefQual l{};
      const NOCRefQual nl{};
      std::same_as<std::optional<int>> decltype(auto) r = i.and_then(l);

      assert(r == 1);
      assert(i.and_then(nl) == std::nullopt);
    }
  }

  // &&
  {
    //With & qualifier on F's operator()
    {
      int j = 42;
      std::optional<int&> i{j};
      std::same_as<std::optional<int>> decltype(auto) r = i.and_then(RVRefQual{});

      assert(r == 1);
      assert(std::move(i).and_then(NORVRefQual{}) == std::nullopt);
    }
  }

  // const&&
  {
    //With & qualifier on F's operator()
    {
      int j = 42;
      const std::optional<int&> i{j};
      const RVCRefQual l{};
      const NORVCRefQual nl{};
      std::same_as<std::optional<int>> decltype(auto) r = std::move(i).and_then(std::move(l));

      assert(r == 1);
      assert(std::move(i).and_then(std::move(nl)) == std::nullopt);
    }
  }
  return true;
}
#endif

int main(int, char**) {
  test();
  static_assert(test());
#if TEST_STD_VER >= 26
  test_ref();
  static_assert(test_ref());
#endif
  return 0;
}
