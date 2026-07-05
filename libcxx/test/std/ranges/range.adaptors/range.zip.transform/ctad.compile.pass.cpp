//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class F, class... Rs>
// zip_transform_view(F, Rs&&...) -> zip_transform_view<F, views::all_t<Rs>...>;

#include <cassert>
#include <ranges>

#include "types.h"

struct Container {
  int* begin() const;
  int* end() const;
};

struct Fn {
  int operator()(auto&&...) const { return 5; }
};

void testCTAD() {
  static_assert(std::is_same_v<decltype(std::ranges::zip_transform_view(Fn{}, Container{})),
                               std::ranges::zip_transform_view<Fn, std::ranges::owning_view<Container>>>);

  static_assert(std::is_same_v<decltype(std::ranges::zip_transform_view(Fn{}, Container{}, IntView{})),
                               std::ranges::zip_transform_view<Fn, std::ranges::owning_view<Container>, IntView>>);

  Container c{};
  static_assert(
      std::is_same_v<
          decltype(std::ranges::zip_transform_view(Fn{}, Container{}, IntView{}, c)),
          std::ranges::
              zip_transform_view<Fn, std::ranges::owning_view<Container>, IntView, std::ranges::ref_view<Container>>>);
}
