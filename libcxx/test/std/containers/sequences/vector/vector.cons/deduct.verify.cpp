//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// UNSUPPORTED: c++03, c++11, c++14

// template <class InputIterator, class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//    vector(InputIterator, InputIterator, Allocator = Allocator())
//    -> vector<typename iterator_traits<InputIterator>::value_type, Allocator>;
//

#include <cassert>
#include <cstddef>
#include <vector>

int main(int, char**) {
  //  Test the explicit deduction guides
  // TODO: Should there be tests for explicit deduction guides?

  //  Test the implicit deduction guides
  {
    //  vector (allocator &)
    // expected-error-re@+1 {{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}vector'}}
    std::vector vec(std::allocator< int>{});
  }

  return 0;
}
