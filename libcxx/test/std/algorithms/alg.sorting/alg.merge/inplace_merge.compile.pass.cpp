//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++20

// inplace_merge used to use ranges::advance, which can result in ambiguous call
// https://cplusplus.github.io/LWG/issue4510

#include <algorithm>
#include <cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct OmniConv {
  OmniConv(const auto&);
  friend bool operator==(OmniConv, OmniConv) = default; // found by ADL via things related to OmniConv

  friend auto operator<=>(OmniConv, OmniConv) = default;
};

void test() {
  OmniConv arr[] = {{1}, {2}, {3}};

  using Iter = bidirectional_iterator<OmniConv*>;

  std::ranges::inplace_merge(Iter(arr), Iter(arr + 1), Iter(arr + 2));
}
