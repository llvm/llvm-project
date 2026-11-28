//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// PROTOTYPE: P3866R0 (V2: An Evolution Path for the Standard Library)
//
// Ways of moving data between std::vector and std::vector2 at API boundaries. None of them is free: they all
// copy (or at best move element-wise and reallocate), since the two types can't share their buffer.

#include <cassert>
#include <iterator>
#include <ranges>
#include <span>
#include <vector>

namespace cpp = std::cpp29;

int sum_legacy(std::vector<int> const& v) {
  int s = 0;
  for (int i : v)
    s += i;
  return s;
}

// The real fix for APIs is to stop taking a concrete container type.
int sum_span(std::span<int const> v) {
  int s = 0;
  for (int i : v)
    s += i;
  return s;
}

int main(int, char**) {
  cpp::vector<int> v2 = {1, 2, 3};

  // Iterator-pair construction
  assert(sum_legacy(std::vector<int>(v2.begin(), v2.end())) == 6);

  // ranges::to
  assert(sum_legacy(std::ranges::to<std::vector<int>>(v2)) == 6);
  auto back = std::ranges::to<cpp::vector<int>>(std::vector<int>{4, 5});
  assert(back.size() == 2);

  // Moving elements doesn't move the buffer
  std::vector<int> v1(std::make_move_iterator(v2.begin()), std::make_move_iterator(v2.end()));
  assert(v1.size() == 3 && v2.size() == 3);

  // Contiguous views work with both, at no cost
  assert(sum_span(v2) == 6);
  assert(sum_span(v1) == 6);

  return 0;
}
