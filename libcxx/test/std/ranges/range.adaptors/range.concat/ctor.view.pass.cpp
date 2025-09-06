//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "types.h"

struct Range : std::ranges::view_base {
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

private:
  int* begin_;
  int* end_;
};

struct MoveAwareView : std::ranges::view_base {
  int moves                 = 0;
  constexpr MoveAwareView() = default;
  constexpr MoveAwareView(MoveAwareView&& other) : moves(other.moves + 1) { other.moves = 1; }
  constexpr MoveAwareView& operator=(MoveAwareView&& other) {
    moves       = other.moves + 1;
    other.moves = 0;
    return *this;
  }
  constexpr const int* begin() const { return &moves; }
  constexpr const int* end() const { return &moves + 1; }
};

constexpr bool test() {
  int buff[]  = {1, 2};
  int buff2[] = {3, 4};

  // constructor from views
  {
    Range range(buff, buff + 2);
    Range range2(buff2, buff2 + 2);
    std::ranges::concat_view view(range, range2);
    auto it  = view.begin();
    auto end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 2);
    assert(*it++ == 3);
    assert(*it++ == 4);
    assert(it == end);
  }

  // Make sure we move the view
  {
    MoveAwareView mv;
    std::ranges::concat_view v{std::move(mv), MoveAwareView{}};
    auto it = v.begin();
    assert(*it++ == 2); // one move from the local variable to parameter, one move from parameter to member
    assert(*it++ == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
