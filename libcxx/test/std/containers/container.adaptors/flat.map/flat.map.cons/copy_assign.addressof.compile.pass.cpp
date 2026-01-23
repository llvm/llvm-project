//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map& operator=(const flat_map& s);

// Validate whether the container can be copy-assigned (move-assigned, swapped)
// with an ADL-hijacking operator&

#include <flat_map>
#include <utility>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  std::flat_map<operator_hijacker, operator_hijacker> so;
  std::flat_map<operator_hijacker, operator_hijacker> s;
  s = so;
  s = std::move(so);
  swap(s, so);
}
