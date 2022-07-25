//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// reverse_iterator

#include <iterator>

#include "test_iterators.h"

int main(int, char**) {
  using BadIter = std::reverse_iterator<forward_iterator<int*>>;
  BadIter i; //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}reverse_iterator<It> requires It to be a bidirectional iterator.}}

  return 0;
}
