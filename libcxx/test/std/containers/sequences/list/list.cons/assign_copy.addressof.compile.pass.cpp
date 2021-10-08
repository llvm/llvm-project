//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Guard the debug iterators against ADL-hijacking.
// XFAIL: LIBCXX-DEBUG-FIXME

// <list>

// list& operator=(const list& c);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <list>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  std::list<operator_hijacker> lo;
  std::list<operator_hijacker> l;
  l = lo;
}
