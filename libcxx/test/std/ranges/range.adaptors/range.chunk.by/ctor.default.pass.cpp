//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// chunk_by_view() requires std::default_initializable<View> &&
//                          std::default_initializable<Pred> = default;

#include <ranges>

#include <cassert>
#include <type_traits>

constexpr int buff[] = {-2, 1, -1, 2};

struct DefaultConstructibleView : std::ranges::view_base {
  DefaultConstructibleView() = default;
  constexpr int const* begin() const { return buff; }
  constexpr int const* end() const { return buff + 4; }
};

struct DefaultConstructiblePredicate {
  DefaultConstructiblePredicate() = default;
  constexpr bool operator()(int x, int y) const { return x != -y; }
};

struct NoDefaultView : std::ranges::view_base {
  NoDefaultView() = delete;
  int* begin() const;
  int* end() const;
};

struct NoDefaultPredicate {
  NoDefaultPredicate() = delete;
  constexpr bool operator()(int, int) const;
};

struct NoexceptView : std::ranges::view_base {
  NoexceptView() noexcept;
  int const* begin() const;
  int const* end() const;
};

struct NoexceptPredicate {
  NoexceptPredicate() noexcept;
  bool operator()(int, int) const;
};

struct MayThrowView : std::ranges::view_base {
  MayThrowView() noexcept(false);
  int const* begin() const;
  int const* end() const;
};

struct MayThrowPredicate {
  MayThrowPredicate() noexcept(false);
  bool operator()(int, int) const;
};

constexpr void compareRanges(std::ranges::subrange<const int*> v, std::initializer_list<int> list) {
  assert(v.size() == list.size());
  for (size_t i = 0; i < v.size(); ++i) {
    assert(v[i] == list.begin()[i]);
  }
}

constexpr bool test() {
  // Check default constructor with default initialization
  {
    using View = std::ranges::chunk_by_view<DefaultConstructibleView, DefaultConstructiblePredicate>;
    View view;
    auto it = view.begin(), end = view.end();
    compareRanges(*it++, {-2, 1});
    compareRanges(*it++, {-1, 2});
    assert(it == end);
  }

  // Check default construction with copy-list-initialization
  {
    using View = std::ranges::chunk_by_view<DefaultConstructibleView, DefaultConstructiblePredicate>;
    View view  = {};
    auto it = view.begin(), end = view.end();
    compareRanges(*it++, {-2, 1});
    compareRanges(*it++, {-1, 2});
    assert(it == end);
  }

  // Check cases where the default constructor isn't provided
  {
    static_assert(
        !std::is_default_constructible_v<std::ranges::chunk_by_view<NoDefaultView, DefaultConstructiblePredicate>>);
    static_assert(
        !std::is_default_constructible_v<std::ranges::chunk_by_view<DefaultConstructibleView, NoDefaultPredicate>>);
    static_assert(!std::is_default_constructible_v<std::ranges::chunk_by_view<NoDefaultView, NoDefaultPredicate>>);
  }

  // Check noexcept-ness
  {
    {
      using View = std::ranges::chunk_by_view<MayThrowView, MayThrowPredicate>;
      static_assert(!noexcept(View()));
    }
    {
      using View = std::ranges::chunk_by_view<MayThrowView, NoexceptPredicate>;
      static_assert(!noexcept(View()));
    }
    {
      using View = std::ranges::chunk_by_view<NoexceptView, MayThrowPredicate>;
      static_assert(!noexcept(View()));
    }
    {
      using View = std::ranges::chunk_by_view<NoexceptView, NoexceptPredicate>;
      static_assert(noexcept(View()));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
