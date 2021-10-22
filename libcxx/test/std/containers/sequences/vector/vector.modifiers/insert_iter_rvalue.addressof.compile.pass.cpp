//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <vector>

// iterator insert(const_iterator position, value_type&& x);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <vector>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  std::vector<operator_hijacker> v;
  v.insert(v.end(), operator_hijacker{});
}
