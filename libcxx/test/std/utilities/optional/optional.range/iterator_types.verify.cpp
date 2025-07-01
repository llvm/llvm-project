//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: std-at-least-c++26

#include <iterator>
#include <optional>

int main() {
  using Iter = std::optional<int>::iterator;

  std::iterator_traits<int> s;
  static_assert(std::random_access_iterator<Iter>);
  static_assert(std::contiguous_iterator<Iter>);

  static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, int>);
  static_assert(std::is_same_v<typename std::iterator_traits<Iter>::difference_type, std::ptrdiff_t>);
  static_assert(std::is_same_v<typename std::iterator_traits<Iter>::pointer, int*>);
  static_assert(std::is_same_v<typename std::iterator_traits<Iter>::reference, int&>);
  static_assert(
      std::is_same_v<typename std::iterator_traits<Iter>::iterator_category, std::random_access_iterator_tag>);
  return 0;
}