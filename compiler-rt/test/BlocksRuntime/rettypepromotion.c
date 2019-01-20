//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *  rettypepromotion.c
 *  testObjects
 *
 *  Created by Blaine Garst on 11/3/08.
 *
 */
 
// CONFIG error:
// C++ and C give different errors so we don't check for an exact match.
// The error is that enum's are defined to be ints, always, even if defined with explicit long values


#include <stdio.h>
#include <stdlib.h>

enum { LESS = -1, EQUAL, GREATER };

void sortWithBlock(long (^comp)(void *arg1, void *arg2)) {
}

int main(int argc, char *argv[]) {
    sortWithBlock(^(void *arg1, void *arg2) {
        if (random()) return LESS;
        if (random()) return EQUAL;
        if (random()) return GREATER;
    });
    printf("%s: Success\n", argv[0]);
    return 0;
}
