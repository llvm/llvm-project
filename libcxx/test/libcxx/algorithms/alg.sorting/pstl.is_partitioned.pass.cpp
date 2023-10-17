//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// Make sure that the predicate is called exactly N times in is_partitioned

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <iterator>

int main(int, char**) {
  std::size_t call_count = 0;
  int a[]        = {1, 2, 3, 4, 5, 6, 7, 8};
  assert(std::is_partitioned(std::execution::seq, std::begin(a), std::end(a), [&](int i) {
    ++call_count;
    return i < 5;
  }));
  assert(call_count == std::size(a));

  return 0;
}
