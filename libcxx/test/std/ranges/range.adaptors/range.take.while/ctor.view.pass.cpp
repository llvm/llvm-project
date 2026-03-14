//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr take_while_view(V base, Pred pred); // explicit since C++23

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"
#include "test_convertible.h"
#include "test_macros.h"

struct View : std::ranges::view_base {
  MoveOnly mo;
  int* begin() const;
  int* end() const;
};

struct Pred {
  bool copied      = false;
  bool moved       = false;
  constexpr Pred() = default;
  constexpr Pred(Pred&&) : moved(true) {}
  constexpr Pred(const Pred&) : copied(true) {}
  bool operator()(int) const;
};

// SFINAE tests.

#if TEST_STD_VER >= 23

static_assert(!test_convertible<std::ranges::take_while_view<View, Pred>, View, Pred>(),
              "This constructor must be explicit");

#else

static_assert(test_convertible<std::ranges::take_while_view<View, Pred>, View, Pred>(),
              "This constructor must not be explicit");

#endif // TEST_STD_VER >= 23

constexpr bool test() {
  {
    std::ranges::take_while_view<View, Pred> twv{View{{}, MoveOnly{5}}, Pred{}};
    assert(twv.pred().moved);
    assert(!twv.pred().copied);
    assert(std::move(twv).base().mo.get() == 5);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
