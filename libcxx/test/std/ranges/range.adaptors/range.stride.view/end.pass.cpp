//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto end() requires(!__simple_view<_View>)
// constexpr auto end() const requires(range<const _View>)

// Note: Checks here are augmented by checks in
// iterator/ctor.copy.pass.cpp.

#include <ranges>

#include "types.h"

template <class T>
concept HasConstEnd = requires(const T& ct) { ct.end(); };

template <class T>
concept HasEnd = requires(T& t) { t.end(); };

template <class T>
concept HasConstAndNonConstEnd =
    HasConstEnd<T> && requires(T& t, const T& ct) { requires !std::same_as<decltype(t.end()), decltype(ct.end())>; };

template <class T>
concept HasOnlyNonConstEnd = HasEnd<T> && !HasConstEnd<T>;

template <class T>
concept HasOnlyConstEnd = HasConstEnd<T> && !HasConstAndNonConstEnd<T>;

struct NoConstView : std::ranges::view_base {
  int* begin();
  int* end();
};

struct UnsimpleConstView : std::ranges::view_base {
  double* begin();
  int* begin() const;

  double* end();
  int* end() const;
};

struct UnsimpleUnCommonView : std::ranges::view_base {
  double* begin();
  char* begin() const;

  void* end();
  void* end() const;
};

struct SimpleUnCommonView : std::ranges::view_base {
  int* begin();
  int* begin() const;

  void* end();
  void* end() const;
};

static_assert(HasOnlyNonConstEnd<std::ranges::stride_view<NoConstView>>);
static_assert(HasOnlyConstEnd<std::ranges::stride_view<SimpleView>>);
static_assert(HasConstAndNonConstEnd<std::ranges::stride_view<UnsimpleConstView>>);

constexpr bool test_non_default_sentinel() {
  {
    LIBCPP_STATIC_ASSERT(std::ranges::__simple_view<SimpleView>);
    static_assert(std::ranges::common_range<SimpleView>);
    static_assert(std::ranges::sized_range<SimpleView>);
    static_assert(std::ranges::forward_range<SimpleView>);

    auto v  = SimpleView{nullptr, nullptr};
    auto sv = std::ranges::stride_view<SimpleView>(v, 1);
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);
  }

  {
    LIBCPP_STATIC_ASSERT(!std::ranges::__simple_view<NoConstView>);
    static_assert(std::ranges::common_range<NoConstView>);
    static_assert(std::ranges::sized_range<NoConstView>);
    static_assert(std::ranges::forward_range<NoConstView>);

    auto v  = NoConstView{};
    auto sv = std::ranges::stride_view<NoConstView>(v, 1);
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);
  }
  return true;
}

constexpr bool test_default_sentinel() {
  {
    LIBCPP_STATIC_ASSERT(!std::ranges::__simple_view<UnsimpleUnCommonView>);
    static_assert(!std::ranges::common_range<UnsimpleUnCommonView>);
    static_assert(std::ranges::sized_range<UnsimpleConstView>);
    static_assert(std::ranges::forward_range<UnsimpleConstView>);

    auto v  = UnsimpleUnCommonView{};
    auto sv = std::ranges::stride_view<UnsimpleUnCommonView>(v, 1);
    static_assert(std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);
  }

  {
    LIBCPP_STATIC_ASSERT(std::ranges::__simple_view<SimpleUnCommonView>);
    static_assert(!std::ranges::common_range<SimpleUnCommonView>);

    auto v  = SimpleUnCommonView{};
    auto sv = std::ranges::stride_view<SimpleUnCommonView>(v, 1);

    static_assert(std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);
  }
  return true;
}

int main(int, char**) {
  test_non_default_sentinel();
  test_default_sentinel();
  static_assert(test_non_default_sentinel());
  static_assert(test_default_sentinel());

  return 0;
}
