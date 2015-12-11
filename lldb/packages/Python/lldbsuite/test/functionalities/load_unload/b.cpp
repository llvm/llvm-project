//===-- b.c -----------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

int b_init()
{
    return 345;
}

int b_global = b_init();

int
b_function ()
{
    return 500;
}
