//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// Test that std::nth_element does not have a quadratic worst case

// UNSUPPORTED: c++03

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace {

std::size_t comparisons = 0;

struct CountingLess {
  bool operator()(int x, int y) const {
    ++comparisons;
    return x < y;
  }
};

// Musser's median-of-3 killer: a permutation on which taking the median of the first, middle and last element as the
// pivot splits off only a constant number of elements per pass.
std::vector<int> median_of_3_killer(int n) {
  std::vector<int> v(n);
  int half = n / 2 + (n % 2);
  for (int i = 1; i <= n; ++i)
    v[i - 1] = (i <= half) ? (i % 2 != 0 ? i : half + i - 1) : 2 * (i - half);
  return v;
}

void test(int n) {
  std::vector<int> v = median_of_3_killer(n);
  std::vector<int> expected(v);
  std::sort(expected.begin(), expected.end());

  std::size_t nth = static_cast<std::size_t>(n) / 2;
  comparisons     = 0;
  std::nth_element(v.begin(), v.begin() + nth, v.end(), CountingLess());

  // Loose: far above what is spent here, far below quadratic.
  assert(comparisons <= 64 * v.size());

  assert(v[nth] == expected[nth]);
  for (std::size_t i = 0; i < nth; ++i)
    assert(!(v[nth] < v[i]));
  for (std::size_t i = nth + 1; i < v.size(); ++i)
    assert(!(v[i] < v[nth]));
}

} // namespace

int main(int, char**) {
  test(1 << 12);
  test(1 << 16);

  return 0;
}
