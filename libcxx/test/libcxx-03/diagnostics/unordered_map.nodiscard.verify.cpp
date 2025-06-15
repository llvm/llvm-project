//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <unordered_map> functions are marked [[nodiscard]]

// clang-format off

#include <unordered_map>

void unordered_map_test() {
  std::unordered_map<int, int> unordered_map;
  unordered_map.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void unordered_multimap_test() {
  std::unordered_multimap<int, int> unordered_multimap;
  unordered_multimap.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
