//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// [Note any is a not a literal type --end note]

#include <any>
#include <type_traits>

#include "test_macros.h"

int main(int, char**) {
    static_assert(!std::is_literal_type<std::any>::value, "");

  return 0;
}
