//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <unordered_set> functions are marked [[nodiscard]]

// clang-format off

#include <unordered_set>

void unordered_set_test() {
  std::unordered_set<int> unordered_set;
  unordered_set.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void unordered_multiset_test() {
  std::unordered_multiset<int> unordered_multiset;
  unordered_multiset.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
