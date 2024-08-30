//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector& operator=(const inplace_vector& c);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <inplace_vector>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  {
    std::inplace_vector<operator_hijacker, 10> vo;
    std::inplace_vector<operator_hijacker, 10> v;
    v = vo;
  }
  {
    std::inplace_vector<operator_hijacker, 0> vo;
    std::inplace_vector<operator_hijacker, 0> v;
    v = vo;
  }
}
