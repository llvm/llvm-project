//===-- c.c -----------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
