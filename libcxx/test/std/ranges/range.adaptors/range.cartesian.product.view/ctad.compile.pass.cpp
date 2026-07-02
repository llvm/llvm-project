//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// template <class... Vs>
// cartesian_product_view(Vs&&...) -> cartesian_product_view<views::all_t<Vs>...>;

#include <ranges>

struct Container {
  int* begin() const;
  int* end() const;
};

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

void testCTAD() {
  // single Container -> owning_view<Container>
  static_assert(std::is_same_v<decltype(std::ranges::cartesian_product_view(Container{})),
                               std::ranges::cartesian_product_view<std::ranges::owning_view<Container>>>);

  // Container + View
  static_assert(std::is_same_v<decltype(std::ranges::cartesian_product_view(Container{}, View{})),
                               std::ranges::cartesian_product_view<std::ranges::owning_view<Container>, View>>);

  // lvalue Container -> ref_view<Container>
  Container c{};
  static_assert(
      std::is_same_v<
          decltype(std::ranges::cartesian_product_view(Container{}, View{}, c)),
          std::ranges::
              cartesian_product_view<std::ranges::owning_view<Container>, View, std::ranges::ref_view<Container>>>);

  // 4-range CTAD with mix of view, lvalue container, rvalue container
  static_assert(std::is_same_v<decltype(std::ranges::cartesian_product_view(View{}, c, Container{}, View{})),
                               std::ranges::cartesian_product_view<View,
                                                                   std::ranges::ref_view<Container>,
                                                                   std::ranges::owning_view<Container>,
                                                                   View>>);

  // CTAD with iota_view and single_view
  static_assert(std::is_same_v<
                decltype(std::ranges::cartesian_product_view(std::views::iota(0, 5), std::ranges::single_view{1})),
                std::ranges::cartesian_product_view<std::ranges::iota_view<int, int>, std::ranges::single_view<int>>>);
}
