//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr filter_view(View, Pred); // explicit since C++23

#include <cassert>
#include <ranges>
#include <utility>

#include "test_convertible.h"
#include "test_macros.h"
#include "types.h"

struct Range : std::ranges::view_base {
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) { }
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

private:
  int* begin_;
  int* end_;
};

struct TrackingRange : TrackInitialization, std::ranges::view_base {
  using TrackInitialization::TrackInitialization;
  int* begin() const;
  int* end() const;
};

constexpr bool test() {
  int buff[] = {1, 2, 3, 4};

  // Test explicit syntax
  {
    Range range(buff, buff + 4);
    std::ranges::concat_view<Range> view(range);
    auto it = view.begin();
    auto end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 2);
    assert(*it++ == 3);
    assert(*it++ == 4);
    assert(it == end);
  }

  // Make sure we move the view
  {
    bool moved = false, copied = false;
    TrackingRange range(&moved, &copied);
    [[maybe_unused]] std::ranges::concat_view<TrackingRange> view(std::move(range));
    assert(moved);
    assert(!copied);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
