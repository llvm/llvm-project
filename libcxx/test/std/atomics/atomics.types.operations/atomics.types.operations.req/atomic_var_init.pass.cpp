//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: c++03
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <atomic>

// #define ATOMIC_VAR_INIT(value)

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::atomic<int> v = ATOMIC_VAR_INIT(5);
    assert(v == 5);

  return 0;
}
