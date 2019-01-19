//===-- c.c -----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

int d_init()
{
    return 123;
}

int d_global = d_init();

int
d_function ()
{ // Find this line number within d_dunction().
    return 700;
}
