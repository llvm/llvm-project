//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::prev

#include <iterator>
#include "test_iterators.h"

void test() {
  int arr[] = {1, 2};
  cpp17_input_iterator<int*> it(&arr[0]);
  it = std::prev(it);
  // expected-error-re@*:* {{static assertion failed due to requirement {{.*}}: Attempt to prev(it) with a non-bidirectional iterator}}
}
