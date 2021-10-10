//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// template <class Iter>
//   iterator insert(const_iterator position, Iter first, Iter last);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <vector>

#include "test_macros.h"
#include "operator_hijacker.h"
#include "test_iterators.h"

void test() {
  {
    std::vector<operator_hijacker> v;
    cpp17_input_iterator<std::vector<operator_hijacker>::iterator> i;
    v.insert(v.end(), i, i);
  }
  {
    std::vector<operator_hijacker> v;
    v.insert(v.end(), v.begin(), v.end());
  }
}
