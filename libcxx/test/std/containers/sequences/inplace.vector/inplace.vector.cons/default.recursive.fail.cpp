//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// Unlike vector, inplace_vector cannot be used with an incomplete type

#include <inplace_vector>

#include "test_macros.h"

struct Y {
  std::inplace_vector<Y, 10> q;
};

int main(int, char**) { return 0; }
