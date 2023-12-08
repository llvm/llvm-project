//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// set& operator=(const set& s);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <set>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  std::set<operator_hijacker> so;
  std::set<operator_hijacker> s;
  s = so;
}
