//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// check that <__node_handle> functions are marked [[nodiscard]]

#include <set>

void func() {
  std::set<int> set;
  set.extract(0).empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
