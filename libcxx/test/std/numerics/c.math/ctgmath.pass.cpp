//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ctgmath>

#include <ctgmath>

int main(int, char**)
{
    std::complex<double> cd;
    (void)cd;
    double x = std::sin(0);
    ((void)x); // Prevent unused warning

  return 0;
}
