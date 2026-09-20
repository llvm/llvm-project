//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <flat_set>

//  template<class Key, class Compare = less<Key>, class KeyContainer = vector<Key>>
//    class flat_set;

#include <flat_set>
#include <functional>
#include <vector>

void test() {
  // expected-error-re@*:* {{static assertion failed{{.*}}The stored elements' key type must match the underlying key container's value_type.}}
  std::flat_set<double, std::less<double>, std::vector<int>> fs1;

  // expected-error-re@*:* {{static assertion failed{{.*}}vector<bool> is not a sequence container}}
  std::flat_set<bool, std::less<bool>, std::vector<bool>> fs2;
}
