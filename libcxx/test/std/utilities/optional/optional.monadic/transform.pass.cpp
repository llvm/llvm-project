//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// XFAIL: availability-bad_optional_access-missing && !no-exceptions

// <optional>

// template<class F> constexpr auto transform(F&&) &;
// template<class F> constexpr auto transform(F&&) &&;
// template<class F> constexpr auto transform(F&&) const&;
// template<class F> constexpr auto transform(F&&) const&&;

#include "test_macros.h"
#include <cassert>
#include <optional>
#include <type_traits>

struct LVal {
  constexpr int operator()(int&) { return 1; }
  int operator()(const int&) = delete;
  int operator()(int&&) = delete;
  int operator()(const int&&) = delete;
};

struct CLVal {
  int operator()(int&) = delete;
  constexpr int operator()(const int&) { return 1; }
  int operator()(int&&) = delete;
  int operator()(const int&&) = delete;
};

struct RVal {
  int operator()(int&) = delete;
  int operator()(const int&) = delete;
  constexpr int operator()(int&&) { return 1; }
  int operator()(const int&&) = delete;
};

struct CRVal {
  int operator()(int&) = delete;
  int operator()(const int&) = delete;
  int operator()(int&&) = delete;
  constexpr int operator()(const int&&) { return 1; }
};

struct RefQual {
  constexpr int operator()(int) & { return 1; }
  int operator()(int) const& = delete;
  int operator()(int) && = delete;
  int operator()(int) const&& = delete;
};

struct CRefQual {
  int operator()(int) & = delete;
  constexpr int operator()(int) const& { return 1; }
  int operator()(int) && = delete;
  int operator()(int) const&& = delete;
};

struct RVRefQual {
  int operator()(int) & = delete;
  int operator()(int) const& = delete;
  constexpr int operator()(int) && { return 1; }
  int operator()(int) const&& = delete;
};

struct RVCRefQual {
  int operator()(int) & = delete;
  int operator()(int) const& = delete;
  int operator()(int) && = delete;
  constexpr int operator()(int) const&& { return 1; }
};

struct NoCopy {
  NoCopy() = default;
  NoCopy(const NoCopy&) { assert(false); }
  int operator()(const NoCopy&&) { return 1; }
};

struct NoMove {
  NoMove() = default;
  NoMove(NoMove&&) = delete;
  NoMove operator()(const NoCopy&&) { return NoMove{}; }
};

constexpr void test_val_types() {
  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      std::optional<int> i{0};
      assert(i.transform(LVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(LVal{})), std::optional<int>);
    }

    //With & qualifier on F's operator()
    {
      std::optional<int> i{0};
      RefQual l{};
      assert(i.transform(l) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), std::optional<int>);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const std::optional<int> i{0};
      assert(i.transform(CLVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(CLVal{})), std::optional<int>);
    }

    //With & qualifier on F's operator()
    {
      const std::optional<int> i{0};
      const CRefQual l{};
      assert(i.transform(l) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), std::optional<int>);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      std::optional<int> i{0};
      assert(std::move(i).transform(RVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(std::move(i).transform(RVal{})), std::optional<int>);
    }

    //With & qualifier on F's operator()
    {
      std::optional<int> i{0};
      assert(i.transform(RVRefQual{}) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(RVRefQual{})), std::optional<int>);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const std::optional<int> i{0};
      assert(std::move(i).transform(CRVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(std::move(i).transform(CRVal{})), std::optional<int>);
    }

    //With & qualifier on F's operator()
    {
      const std::optional<int> i{0};
      const RVCRefQual l{};
      assert(i.transform(std::move(l)) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(std::move(l))), std::optional<int>);
    }
  }
}

struct NonConst {
  int non_const() { return 1; }
};

// check that the lambda body is not instantiated during overload resolution
constexpr void test_sfinae() {
  std::optional<NonConst> opt{};
  auto l = [](auto&& x) { return x.non_const(); };
  opt.transform(l);
  std::move(opt).transform(l);
}

constexpr bool test() {
  test_sfinae();
  test_val_types();
  std::optional<int> opt;
  const auto& copt = opt;

  const auto never_called = [](int) {
    assert(false);
    return 0;
  };

  opt.transform(never_called);
  std::move(opt).transform(never_called);
  copt.transform(never_called);
  std::move(copt).transform(never_called);

  std::optional<NoCopy> nc;
  const auto& cnc = nc;
  std::move(nc).transform(NoCopy{});
  std::move(cnc).transform(NoCopy{});

  std::move(nc).transform(NoMove{});
  std::move(cnc).transform(NoMove{});

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
