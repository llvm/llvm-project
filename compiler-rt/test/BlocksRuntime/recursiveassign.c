//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *  recursiveassign.c
 *  testObjects
 *
 *  Created by Blaine Garst on 12/3/08.
 *
 */

// CONFIG  rdar://6639533

// The compiler is prefetching x->forwarding before evaluating code that recomputes forwarding and so the value goes to a place that is never seen again.

#include <stdio.h>
#include <stdlib.h>
#include <Block.h>


int main(int argc, char* argv[]) {
    
    __block void (^recursive_copy_block)(int) = ^(int arg) { printf("got wrong Block\n"); exit(1); };
    
    
    recursive_copy_block = Block_copy(^(int i) {
        if (i > 0) {
            recursive_copy_block(i - 1);
        }
        else {
            printf("done!\n");
        }
    });
    
    
    recursive_copy_block(5);
    
    printf("%s: Success\n", argv[0]);
    return 0;
}

