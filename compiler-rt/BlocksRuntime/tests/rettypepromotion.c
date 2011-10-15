//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

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
