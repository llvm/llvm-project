//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <optional>

// template<class F> constexpr auto transform(F&&) &;
// template<class F> constexpr auto transform(F&&) &&;
// template<class F> constexpr auto transform(F&&) const&;
// template<class F> constexpr auto transform(F&&) const&&;

#include "test_macros.h"
#include <cassert>
#include <concepts>
#include <optional>
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

struct NoCopy {
  NoCopy() = default;
  NoCopy(const NoCopy&) { assert(false); }
  int operator()(const NoCopy&&) { return 1; }
};

struct NoMove {
  NoMove()         = default;
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
  (void)opt.transform(l);
  (void)std::move(opt).transform(l);
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

  (void)opt.transform(never_called);
  (void)std::move(opt).transform(never_called);
  (void)copt.transform(never_called);
  (void)std::move(copt).transform(never_called);

  std::optional<NoCopy> nc;
  const auto& cnc = nc;
  (void)std::move(nc).transform(NoCopy{});
  (void)std::move(cnc).transform(NoCopy{});

  (void)std::move(nc).transform(NoMove{});
  (void)std::move(cnc).transform(NoMove{});

  return true;
}

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  // Test that no matter the ref qualifier on the object .transform() is invoked on, only the added
  // const (no ref-qualifier) overload is used
  {
    std::optional<int&> opt1;
    std::same_as<std::optional<int>> decltype(auto) opt1r = opt1.transform([](int i) { return i + 2; });
    assert(!opt1);
    assert(!opt1r);
  }

  {
    int i = 42;
    std::optional<int&> opt{i};
    std::same_as<std::optional<int>> decltype(auto) o2 = opt.transform([](int j) { return j + 2; });

    assert(*o2 == 44);
  }

  {
    int i   = 42;
    float k = 4.0f;
    std::optional<int&> opt{i};
    std::same_as<std::optional<float>> decltype(auto) o2 = opt.transform([&](int&) { return k; });
    assert(*o2 == 4.0f);
  }

  // &
  {
    // Without & qualifier on F's operator()
    {
      int i = 42;
      std::optional<int&> opt{i};
      std::same_as<std::optional<int>> decltype(auto) o3 = opt.transform(LVal{});

      assert(*o3 == 1);
    }

    //With & qualifier on F's operator()
    {
      int i = 42;
      std::optional<int&> opt{i};
      RefQual l{};
      std::same_as<std::optional<int>> decltype(auto) o3 = opt.transform(l);

      assert(*o3 == 1);
    }
  }
  // const& overload
  {
    // Without & qualifier on F's operator()
    {
      int i = 42;
      const std::optional<const int&> opt{i};
      std::same_as<std::optional<int>> decltype(auto) o3 = std::as_const(opt).transform(CLVal{});

      assert(*o3 == 1);
    }

    //With & qualifier on F's operator()
    {
      int i = 42;
      const std::optional<int&> opt{i};
      const CRefQual l{};
      std::same_as<std::optional<int>> decltype(auto) o3 = opt.transform(l);

      assert(*o3 == 1);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      int i = 42;
      std::optional<int&> opt{i};
      std::same_as<std::optional<int>> decltype(auto) o3 = std::move(opt).transform(LVal{});

      assert(*o3 == 1);
    }

    //With & qualifier on F's operator()
    {
      int i = 42;
      std::optional<int&> opt{i};
      std::same_as<std::optional<int>> decltype(auto) o3 = std::move(opt).transform(RVRefQual{});
      assert(*o3 == 1);
    }
  }

  // const&& overload
  {
    //With & qualifier on F's operator()
    {
      int i = 42;
      const std::optional<int&> opt{i};
      const RVCRefQual rvc{};
      std::same_as<std::optional<int>> decltype(auto) o3 = std::move(opt).transform(std::move(rvc));
      assert(*o3 == 1);
    }
  }
  {
    std::optional<int&> o6 = std::nullopt;
    auto o6r               = o6.transform([](int) { return 42; });
    assert(!o6r);
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
