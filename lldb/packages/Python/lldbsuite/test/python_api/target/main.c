//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

// This simple program is to test the lldb Python API SBTarget.
//
// When stopped on breakppint 1, and then 2, we can get the line entries using
// SBFrame API SBFrame.GetLineEntry().  We'll get the start addresses for the
// two line entries; with the start address (of SBAddress type), we can then
// resolve the symbol context using the SBTarget API
// SBTarget.ResolveSymbolContextForAddress().
//
// The two symbol context should point to the same symbol, i.e., 'a' function.

char my_global_var_of_char_type = 'X'; // Test SBTarget.FindGlobalVariables(...).

int a(int);
int b(int);
int c(int);

int a(int val)
{
    if (val <= 1) // Find the line number for breakpoint 1 here.
        val = b(val);
    else if (val >= 3)
        val = c(val);

    return val; // Find the line number for breakpoint 2 here.
}

int b(int val)
{
    return c(val);
}

int c(int val)
{
    return val + 3;
}

int main (int argc, char const *argv[])
{
    // Set a break at entry to main.
    int A1 = a(1);  // a(1) -> b(1) -> c(1)
    printf("a(1) returns %d\n", A1);
    
    int B2 = b(2);  // b(2) -> c(2)
    printf("b(2) returns %d\n", B2);
    
    int A3 = a(3);  // a(3) -> c(3)
    printf("a(3) returns %d\n", A3);
    
    return 0;
}
