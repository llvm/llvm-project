//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CONFIG rdar://6255170

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <Block.h>
#include <Block_private.h>
#include <assert.h>


int
main(int argc, char *argv[])
{
    __block int var = 0;
    int shouldbe = 0;
    void (^b)(void) = ^{ var++; /*printf("var is at %p with value %d\n", &var, var);*/ };
    __typeof(b) _b;
    //printf("before copy...\n");
    b(); ++shouldbe;
    size_t i;

    for (i = 0; i < 10; i++) {
            _b = Block_copy(b); // make a new copy each time
            assert(_b);
            ++shouldbe;
            _b();               // should still update the stack
            Block_release(_b);
    }

    //printf("after...\n");
    b(); ++shouldbe;

    if (var != shouldbe) {
        printf("Hmm, var is %d but should be %d\n", var, shouldbe);
        return 1;
    }
    printf("%s: Success!!\n", argv[0]);

    return 0;
}
