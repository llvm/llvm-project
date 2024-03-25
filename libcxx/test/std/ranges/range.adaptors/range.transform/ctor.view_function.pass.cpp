//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr transform_view(View, F); // explicit since C++23

#include <cassert>
#include <ranges>

#include "test_convertible.h"
#include "test_macros.h"

struct Range : std::ranges::view_base {
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

private:
  int* begin_;
  int* end_;
};

struct F {
  constexpr int operator()(int i) const { return i + 100; }
};

// SFINAE tests.

#if TEST_STD_VER >= 23

static_assert(!test_convertible<std::ranges::transform_view<Range, F>, Range, F>(),
              "This constructor must be explicit");

#else

static_assert( test_convertible<std::ranges::transform_view<Range, F>, Range, F>(),
              "This constructor must not be explicit");

#endif // TEST_STD_VER >= 23

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

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
