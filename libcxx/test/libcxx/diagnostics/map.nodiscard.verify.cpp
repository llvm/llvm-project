//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <map> functions are marked [[nodiscard]]

#include <map>

void map_test() {
  std::map<int, int> map;
  map.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void multimap_test() {
  std::multimap<int, int> multimap;
  multimap.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
