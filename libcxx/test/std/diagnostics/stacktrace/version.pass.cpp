//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <stacktrace>

#include <stacktrace>
#include <version>

#include "test_macros.h"

#ifndef __cpp_lib_stacktrace
#  error "__cpp_lib_stacktrace is not defined"
#endif

#if __cpp_lib_stacktrace < 202011L
#  error "__cpp_lib_stacktrace has an invalid value"
#endif

int main(int, char**) {
  return 0;
}
