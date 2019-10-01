//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tgmath.h>

#include <tgmath.h>

#ifndef _LIBCPP_VERSION
#error _LIBCPP_VERSION not defined
#endif

int main(int, char**)
{
    std::complex<double> cd;
    (void)cd;
    double x = sin(1.0);
    (void)x; // to placate scan-build

  return 0;
}
