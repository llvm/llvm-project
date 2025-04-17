//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template <class... Vs>
// cartesian_product_view(Vs&&...) -> cartesian_product_view<views::all_t<Vs>...>;

#include <cassert>
#include <ranges>
#include <utility>

struct Container {
  int* begin() const;
  int* end() const;
};

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

void testCTAD() {
  static_assert(std::is_same_v<decltype(std::ranges::cartesian_product_view(Container{})),
                               std::ranges::cartesian_product_view<std::ranges::owning_view<Container>>>);

  static_assert(std::is_same_v<decltype(std::ranges::cartesian_product_view(Container{}, View{})),
                               std::ranges::cartesian_product_view<std::ranges::owning_view<Container>, View>>);

  Container c{};
  static_assert(
      std::is_same_v<
          decltype(std::ranges::cartesian_product_view(Container{}, View{}, c)),
          std::ranges::
              cartesian_product_view<std::ranges::owning_view<Container>, View, std::ranges::ref_view<Container>>>);
}
