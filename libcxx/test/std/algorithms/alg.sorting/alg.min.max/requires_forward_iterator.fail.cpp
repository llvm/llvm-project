//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   max_element(Iter first, Iter last);

#include <algorithm>
#include <cassert>

#include "test_iterators.h"

int main() {
  int arr[] = {1, 2, 3};
  const int *b = std::begin(arr), *e = std::end(arr);
  typedef input_iterator<const int*> Iter;
  {
    // expected-error@algorithm:* {{"std::min_element requires a ForwardIterator"}}
    std::min_element(Iter(b), Iter(e));
  }
  {
    // expected-error@algorithm:* {{"std::max_element requires a ForwardIterator"}}
    std::max_element(Iter(b), Iter(e));
  }
  {
    // expected-error@algorithm:* {{"std::minmax_element requires a ForwardIterator"}}
    std::minmax_element(Iter(b), Iter(e));
  }

}
