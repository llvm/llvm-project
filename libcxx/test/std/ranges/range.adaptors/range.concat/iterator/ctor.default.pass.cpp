//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-fblocks
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// constexpr iterator() = default;

#include <ranges>
#include <type_traits>

#include "test_macros.h"

struct IntView : std::ranges::view_base {
  int* b_ = nullptr;
  int* e_ = nullptr;

  constexpr IntView() = default;
  constexpr IntView(int* b, int* e) : b_(b), e_(e) {}

  constexpr int* begin() const { return b_; }
  constexpr int* end() const { return e_; }
};

static_assert(std::ranges::view<IntView>);
static_assert(std::ranges::contiguous_range<IntView>);

constexpr bool test() {
  int buf1[] = {1, 2};
  int buf2[] = {3, 4};

  std::ranges::concat_view<IntView, IntView> v(IntView{buf1, buf1 + 2}, IntView{buf2, buf2 + 2});
  using Iter = std::ranges::iterator_t<decltype(v)>;
  static_assert(std::default_initializable<Iter>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
