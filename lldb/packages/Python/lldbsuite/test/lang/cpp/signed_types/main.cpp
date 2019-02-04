//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

int main (int argc, char const *argv[])
{
    char the_char = 'c';
    short the_short = 'c';
    wchar_t the_wchar_t = 'c';
    int the_int = 'c';
    long the_long = 'c';
    long long the_long_long = 'c';

    signed char the_signed_char = 'c';
    signed short the_signed_short = 'c';
    signed int the_signed_int = 'c';
    signed long the_signed_long = 'c';
    signed long long the_signed_long_long = 'c';
    puts("");    // Set break point at this line.
    return  the_char        - the_signed_char +
            the_short       - the_signed_short +
            the_int         - the_signed_int +
            the_long        - the_signed_long +
            the_long_long   - the_signed_long_long; //// break $source:$line; c
    //// var the_int
    //// val -set 22 1
}
