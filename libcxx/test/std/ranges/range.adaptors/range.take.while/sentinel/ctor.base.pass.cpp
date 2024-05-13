//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr explicit sentinel(sentinel_t<Base> end, const Pred* pred);

#include <cassert>
#include <ranges>
#include <utility>

#include "../types.h"

struct Sent {
  int i;

  friend constexpr bool operator==(int* iter, const Sent& s) { return s.i > *iter; }
};

struct Range : std::ranges::view_base {
  int* begin() const;
  Sent end();
};

struct Pred {
  bool operator()(int i) const;
};

// Test explicit
template <class T>
void conversion_test(T);

template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({std::forward<Args>(args)...}); };
static_assert(ImplicitlyConstructible<int, int>);

static_assert(std::is_constructible_v<std::ranges::sentinel_t<std::ranges::take_while_view<Range, Pred>>,
                                      std::ranges::sentinel_t<Range>,
                                      const Pred*>);
static_assert(!ImplicitlyConstructible<std::ranges::sentinel_t<std::ranges::take_while_view<Range, Pred>>,
                                       std::ranges::sentinel_t<Range>,
                                       const Pred*>);

constexpr bool test() {
  // base is init correctly
  {
    using R        = std::ranges::take_while_view<Range, bool (*)(int)>;
    using Sentinel = std::ranges::sentinel_t<R>;

    Sentinel s1(Sent{5}, nullptr);
    assert(s1.base().i == 5);
  }

  // pred is init correctly
  {
    bool called = false;
    auto pred   = [&](int) {
      called = true;
      return false;
    };

    using R        = std::ranges::take_while_view<Range, decltype(pred)>;
    using Sentinel = std::ranges::sentinel_t<R>;

    int i     = 10;
    int* iter = &i;
    Sentinel s(Sent{0}, &pred);

    bool b = iter == s;
    assert(called);
    assert(b);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
