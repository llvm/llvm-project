//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <mutex>

// class mutex;

// mutex();

#include <mutex>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert(std::is_nothrow_default_constructible<std::mutex>::value, "");
    std::mutex m;

  return 0;
}
