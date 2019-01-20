//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *  structmember.c
 *  testObjects
 *
 *  Created by Blaine Garst on 9/30/08.
 *  CONFIG
 */
#include <Block.h>
#include <Block_private.h>
#include <stdio.h>

// CONFIG

int main(int argc, char *argv[]) {
    struct stuff {
        long int a;
        long int b;
        long int c;
    } localStuff = { 10, 20, 30 };
    int d;
    
    void (^a)(void) = ^ { printf("d is %d", d); };
    void (^b)(void) = ^ { printf("d is %d, localStuff.a is %lu", d, localStuff.a); };

    unsigned nominalsize = Block_size(b) - Block_size(a);
#if __cplusplus__
    // need copy+dispose helper for C++ structures
    nominalsize += 2*sizeof(void*);
#endif
    if ((Block_size(b) - Block_size(a)) != nominalsize) {
        printf("sizeof a is %ld, sizeof b is %ld, expected %d\n", Block_size(a), Block_size(b), nominalsize);
        printf("dump of b is %s\n", _Block_dump(b));
        return 1;
    }
    printf("%s: Success\n", argv[0]);
    return 0;
}


