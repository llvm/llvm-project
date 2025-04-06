//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ccomplex> // deprecated in C++17, removed in C++20, but still provided by libc++ as an extension

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <ccomplex>

#include "test_macros.h"

int main(int, char**)
{
    std::complex<double> d;
    (void)d;

  return 0;
}
