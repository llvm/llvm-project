//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// map& operator=(const map& m);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <map>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  {
    std::map<int, operator_hijacker> mo;
    std::map<int, operator_hijacker> m;
    m = mo;
  }
  {
    std::map<operator_hijacker, int> mo;
    std::map<operator_hijacker, int> m;
    m = mo;
  }
}
