//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// __capacity_aware_iterator<class _Iter, class _Tag, std::size_t>

// Verify that only contiguous_iterators are accepted

#include <__iterator/capacity_aware_iterator.h>

#include "test_iterators.h"

void test() {
  std::__capacity_aware_iterator<cpp20_random_access_iterator<int*>, int[], 0>
      v1; // expected-error@*:* {{static assertion failed: __capacity_aware_iterator can only be used with contiguous iterators}}
  std::__capacity_aware_iterator<cpp20_input_iterator<int*>, int[], 0>
      v2; // expected-error@*:* {{static assertion failed: __capacity_aware_iterator can only be used with contiguous iterators}}
  std::__capacity_aware_iterator<bidirectional_iterator<int*>, int[], 0>
      v3; // expected-error@*:* {{static assertion failed: __capacity_aware_iterator can only be used with contiguous iterators}}
  std::__capacity_aware_iterator<cpp20_output_iterator<int*>, int[], 0>
      v4; // expected-error@*:* {{static assertion failed: __capacity_aware_iterator can only be used with contiguous iterators}}
  std::__capacity_aware_iterator<forward_iterator<int*>, int[], 0>
      v5; // expected-error@*:* {{static assertion failed: __capacity_aware_iterator can only be used with contiguous iterators}}
}
