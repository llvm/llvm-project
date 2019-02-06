//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

int main (int argc, char const *argv[])
{
    enum days {
        Monday = 10,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday,
        kNumDays
    };
    enum days day;
    for (day = Monday - 1; day <= kNumDays + 1; day++)
    {
        printf("day as int is %i\n", (int)day);
    }
    return 0; // Set break point at this line.
}
