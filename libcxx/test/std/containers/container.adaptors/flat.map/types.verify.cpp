//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <flat_map>

//  template<class Key, class T, class Compare = less<Key>,
//           class KeyContainer = vector<Key>, class MappedContainer = vector<T>>
//    class flat_map;

#include <flat_map>
#include <functional>
#include <vector>

void test() {
  // expected-error-re@*:* {{static assertion failed{{.*}}The stored elements' key type must match the underlying key container's value_type.}}
  std::flat_map<double, int, std::less<double>, std::vector<char>> fm1;

  // expected-error-re@*:* {{static assertion failed{{.*}}The stored elements' mapped type must match the underlying mapped container's value_type.}}
  std::flat_map<int, double, std::less<int>, std::vector<int>, std::vector<char>> fm2;

  // expected-error-re@*:* {{static assertion failed{{.*}}vector<bool> is not a sequence container}}
  std::flat_map<bool, int, std::less<bool>, std::vector<bool>> fm3;

  // expected-error-re@*:* {{static assertion failed{{.*}}vector<bool> is not a sequence container}}
  std::flat_map<int, bool, std::less<int>, std::vector<int>, std::vector<bool>> fm4;
}
