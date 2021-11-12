//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <forward_list>

// template <class Compare> void merge(forward_list&& x, Compare comp);

// Validate whether the operation properly guards against ADL-hijacking operator&

#include <forward_list>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  std::forward_list<operator_hijacker> lo;
  std::forward_list<operator_hijacker> l;
  lo.merge(std::move(l), std::less<operator_hijacker>());
}
