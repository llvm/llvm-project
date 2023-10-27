//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <set>

// class set

// template<class Key, class Compare, class Allocator>
//   synth-three-way-result<Key> operator<=>(const set<Key, Compare, Allocator>& x,
//                                           const set<Key, Compare, Allocator>& y);

#include <set>

#include "test_allocator.h"

int main(int, char**) {
  // Mismatching allocators
  {
    std::multiset<int, std::less<int>, std::allocator<int>> s1;
    std::multiset<int, std::less<int>, test_allocator<int>> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s1 <=> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s2 <=> s1;
  }
  // Mismatching comparision functions
  {
    std::multiset<int, std::less<int>> s1;
    std::multiset<int, std::greater<int>> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s1 <=> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s2 <=> s1;
  }
  {
    std::multiset<int, std::less<int>> s1;
    std::multiset<int, std::less<float>> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s1 <=> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s2 <=> s1;
  }
  // Mismatching types
  {
    std::multiset<int> s1;
    std::multiset<float> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s1 <=> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s2 <=> s1;
  }

  return 0;
}
