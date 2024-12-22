//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// filter_view() requires std::default_initializable<View> &&
//                        std::default_initializable<Pred> = default;

#include <ranges>

#include <cassert>
#include <type_traits>

constexpr int buff[] = {1, 2, 3, 4};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + 4) {}
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }

private:
  int const* begin_;
  int const* end_;
};

struct NoDefaultView : std::ranges::view_base {
  NoDefaultView() = delete;
  int* begin() const;
  int* end() const;
};

struct NoexceptView : std::ranges::view_base {
  NoexceptView() noexcept;
  int const* begin() const;
  int const* end() const;
};

constexpr bool test() {
  {
    using View = std::ranges::concat_view<DefaultConstructibleView>;
    View view;
    auto it  = view.begin();
    auto end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 2);
    assert(*it++ == 3);
    assert(*it++ == 4);
    assert(it == end);
  }

  // Check cases where the default constructor isn't provided
  {
    static_assert(!std::is_default_constructible_v<std::ranges::concat_view<NoDefaultView >>);
  }

  // Check noexcept-ness
  {
    {
      using View = std::ranges::concat_view<DefaultConstructibleView>;
      static_assert(!noexcept(View()));
    }
    {
      using View = std::ranges::concat_view<NoexceptView>;
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
