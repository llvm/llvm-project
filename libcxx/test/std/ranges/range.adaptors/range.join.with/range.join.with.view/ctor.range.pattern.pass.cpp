//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// constexpr explicit join_with_view(V base, Pattern pattern);

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <utility>

#include "../types.h"

class View : public std::ranges::view_base {
  using OuterRange = std::array<std::array<int, 2>, 3>;

  static constexpr OuterRange default_range = {{{1, 2}, {3, 4}, {5, 6}}};
  static constexpr OuterRange range_on_move = {{{6, 5}, {4, 3}, {2, 1}}};

  const OuterRange* r_ = &default_range;

public:
  View() = default;
  constexpr View(const View&) : r_(&default_range) {}
  constexpr View(View&&) : r_(&range_on_move) {}

  constexpr View& operator=(View) {
    r_ = &default_range;
    return *this;
  }

  constexpr auto begin() { return r_->begin(); }
  constexpr auto end() { return r_->end(); }
};

class Pattern : public std::ranges::view_base {
  using PatternRange = std::array<int, 2>;

  static constexpr PatternRange default_range = {0, 0};
  static constexpr PatternRange range_on_move = {7, 7};

  const PatternRange* val_ = &default_range;

public:
  Pattern() = default;
  constexpr Pattern(const Pattern&) : val_(&default_range) {}
  constexpr Pattern(Pattern&&) : val_(&range_on_move) {}

  constexpr Pattern& operator=(Pattern) {
    val_ = &default_range;
    return *this;
  }

  constexpr auto begin() { return val_->begin(); }
  constexpr auto end() { return val_->end(); }
};

constexpr bool test() {
  {   // Check construction from `view` and `pattern`
    { // `view` and `pattern` are glvalues
      View v;
      Pattern p;
      std::ranges::join_with_view<View, Pattern> jwv(v, p);
      assert(std::ranges::equal(jwv, std::array{6, 5, 7, 7, 4, 3, 7, 7, 2, 1}));
    }

    { // `view` and `pattern` are const glvalues
      const View v;
      const Pattern p;
      std::ranges::join_with_view<View, Pattern> jwv(v, p);
      assert(std::ranges::equal(jwv, std::array{6, 5, 7, 7, 4, 3, 7, 7, 2, 1}));
    }

    { // `view` and `pattern` are prvalues
      std::ranges::join_with_view<View, Pattern> jwv(View{}, Pattern{});
      assert(std::ranges::equal(jwv, std::array{6, 5, 7, 7, 4, 3, 7, 7, 2, 1}));
    }

    { // `view` and `pattern` are xvalues
      View v;
      Pattern p;
      std::ranges::join_with_view<View, Pattern> jwv(std::move(v), std::move(p));
      assert(std::ranges::equal(jwv, std::array{6, 5, 7, 7, 4, 3, 7, 7, 2, 1}));
    }
  }

  // Check explicitness
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, View, Pattern>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, View&, Pattern&>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, const View, const Pattern>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, const View&, const Pattern&>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
