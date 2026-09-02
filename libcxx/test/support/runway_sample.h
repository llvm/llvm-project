//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef TEST_SUPPORT_RUNWAY_SAMPLE_H
#define TEST_SUPPORT_RUNWAY_SAMPLE_H

#include <cstddef>
#include <algorithm>
#include "test_macros.h"

// runway_sample(size, callable) calls callable(i) for a set of sample indices i in the range [0, size).
// they are chosen in 3 phases:
//   - prefix, sampled densely with increments of 1
//   - middle, sampled sparsely with uncanny numbers
//   - suffix, sampled densely with increments of 1
//
// Examples of sample indices for various sizes:
// size = 0:
// <no samples>
// size = 1:
// 0
// size = 10:
// 0 1 2 3 4 5 6 7 8 9
// size = 100:
// 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 50 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99
// size = 1000:
// 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 50 157 493 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999
template <class Callable>
TEST_CONSTEXPR_CXX14 void runway_sample(std::size_t size, Callable callable) {
  constexpr std::size_t affix = 16;
  std::size_t i               = 0;

  // 0, 1, 2, ..., 15
  for (; i < std::min(size, affix); ++i) {
    callable(i);
  }

  if (size <= affix) {
    return;
  }

  // 16, 50, 157, 493, 1549, 4868, ...
  for (std::size_t j = i; j + affix < size; i = j, j = j * 3 + j / 7) {
    callable(j);
  }

  // size - 16, size - 15, ..., size - 1
  i = std::max(i, size - affix);
  for (; i < size; ++i) {
    callable(i);
  }
}

#endif // TEST_SUPPORT_RUNWAY_SAMPLE_H
