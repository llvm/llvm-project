//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr transform_view(View, F);

#include <ranges>

#include <cassert>

struct Range : std::ranges::view_base {
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) { }
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

private:
  int* begin_;
  int* end_;
};

struct F {
  constexpr int operator()(int i) const { return i + 100; }
};

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    Range range(buff, buff + 8);
    F f;
    std::ranges::transform_view<Range, F> view(range, f);
    assert(view[0] == 101);
    assert(view[1] == 102);
    // ...
    assert(view[7] == 108);
  }

  {
    Range range(buff, buff + 8);
    F f;
    std::ranges::transform_view<Range, F> view = {range, f};
    assert(view[0] == 101);
    assert(view[1] == 102);
    // ...
    assert(view[7] == 108);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
