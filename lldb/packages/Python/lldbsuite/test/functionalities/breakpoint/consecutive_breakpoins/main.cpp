//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

int
main(int argc, char const *argv[])
{
    int a = 0;
    int b = 1;
    a = b + 1; // Set breakpoint here
    b = a + 1;
    return 0;
}

