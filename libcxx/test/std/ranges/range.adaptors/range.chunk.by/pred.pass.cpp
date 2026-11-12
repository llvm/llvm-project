//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// constexpr Pred const& pred() const;

#include <ranges>

#include <cassert>
#include <concepts>

struct Range : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct Pred {
  bool operator()(int, int) const;
  int value;
};

constexpr bool test() {
  {
    Pred pred{42};
    std::ranges::chunk_by_view<Range, Pred> const view(Range{}, pred);
    std::same_as<Pred const&> decltype(auto) result = view.pred();
    assert(result.value == 42);

    // Make sure we're really holding a reference to something inside the view
    assert(&result == &view.pred());
  }

  // Same, but calling on a non-const view
  {
    Pred pred{42};
    std::ranges::chunk_by_view<Range, Pred> view(Range{}, pred);
    std::same_as<Pred const&> decltype(auto) result = view.pred();
    assert(result.value == 42);

    // Make sure we're really holding a reference to something inside the view
    assert(&result == &view.pred());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
