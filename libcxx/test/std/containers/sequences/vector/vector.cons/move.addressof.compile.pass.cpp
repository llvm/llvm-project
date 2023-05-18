//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03 && !stdlib=libc++

// <vector>

// vector(vector&& c);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <vector>
#include <utility>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  {
    std::vector<operator_hijacker> vo;
    std::vector<operator_hijacker> v(std::move(vo));
  }
  {
    std::vector<operator_hijacker> vo;
    std::vector<operator_hijacker> v(std::move(vo), std::allocator<operator_hijacker>());
  }
}
