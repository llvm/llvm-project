//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

#include <stdio.h>

// Forward declare an enumeration (only works in C, not C++)
typedef enum ops ops;

struct foo {
	ops *op;
};

int main (int argc, char const *argv[])
{
    enum days {
        Monday = -3,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday,
        kNumDays
    };
    enum days day;
    struct foo f;
    f.op = NULL;
    for (day = Monday - 1; day <= kNumDays + 1; day++)
    {
        printf("day as int is %i\n", (int)day); // Set break point at this line.
    }
    return 0;
}
