//===-- divmod.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

int
main(int argc, char const *argv[])
{
    signed long long a = 123456789, b = 12, c = a / b, d = a % b;
    unsigned long long e = 123456789, f = 12, g = e / f, h = e % f;
}
