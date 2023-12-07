//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <map>

// class multimap

// template<class Key, class T, class Compare, class Allocator>
//   synth-three-way-result<pair<const Key, T>>
//     operator<=>(const multimap<Key, T, Compare, Allocator>& x,
//                 const multimap<Key, T, Compare, Allocator>& y);

#include <map>

#include "test_allocator.h"

int main(int, char**) {
  // Mismatching allocators
  {
    std::multimap<int, int, std::less<int>, std::allocator<int>> s1;
    std::multimap<int, int, std::less<int>, test_allocator<int>> s2;
    // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed due to requirement 'is_same<int, std::pair<const int, int>>::value'{{.*}}Allocator::value_type must be same type as value_type}}
    s1 <=> s2;
    // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed due to requirement 'is_same<int, std::pair<const int, int>>::value'{{.*}}Allocator::value_type must be same type as value_type}}
    s2 <=> s1;
  }
  // Mismatching comparision functions
  {
    std::multimap<int, int, std::less<int>> s1;
    std::multimap<int, int, std::greater<int>> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s1 <=> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s2 <=> s1;
  }
  {
    std::multimap<int, int, std::less<int>> s1;
    std::multimap<int, int, std::less<float>> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s1 <=> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s2 <=> s1;
  }
  // Mismatching types
  {
    std::multimap<int, int> s1;
    std::multimap<int, float> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s1 <=> s2;
    // expected-error@+1 {{invalid operands to binary expression}}
    s2 <=> s1;
  }

  return 0;
}
