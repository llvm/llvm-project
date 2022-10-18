//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

//   template<class R, class Pred>
//     drop_while_view(R&&, Pred) -> drop_while_view<views::all_t<R>, Pred>;

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

struct Pred {
  bool operator()(int i) const;
};

bool pred(int);

static_assert(std::is_same_v<decltype(std::ranges::drop_while_view(Container{}, Pred{})),
                             std::ranges::drop_while_view<std::ranges::owning_view<Container>, Pred>>);

static_assert(std::is_same_v<decltype(std::ranges::drop_while_view(View{}, pred)), //
                             std::ranges::drop_while_view<View, bool (*)(int)>>);

static_assert(std::is_same_v<decltype(std::ranges::drop_while_view(View{}, Pred{})), //
                             std::ranges::drop_while_view<View, Pred>>);

void testRef() {
  Container c{};
  Pred p{};
  static_assert(std::is_same_v<decltype(std::ranges::drop_while_view(c, p)),
                               std::ranges::drop_while_view<std::ranges::ref_view<Container>, Pred>>);
}
