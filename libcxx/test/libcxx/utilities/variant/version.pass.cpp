//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <variant>

// GCC 5 pretends it supports C++17, but it doesn't properly support it, and
// <variant> fails.
// UNSUPPORTED: gcc-5

#include <variant>

#include "test_macros.h"

#ifndef _LIBCPP_VERSION
#error _LIBCPP_VERSION not defined
#endif

int main(int, char**)
{

  return 0;
}
