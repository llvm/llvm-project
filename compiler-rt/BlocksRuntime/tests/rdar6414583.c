//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// CONFIG rdar://6414583

// a smaller case of byrefcopyint

#include <Block.h>
#include <dispatch/dispatch.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    __block int c = 1;

    //printf("&c = %p - c = %i\n", &c, c);

    int i;
    for(i =0; i < 2; i++) {
        dispatch_block_t block = Block_copy(^{ c = i; });

        block();
//        printf("%i: &c = %p - c = %i\n", i, &c, c);

        Block_release(block);
    }
    printf("%s: success\n", argv[0]);
    return 0;
}
