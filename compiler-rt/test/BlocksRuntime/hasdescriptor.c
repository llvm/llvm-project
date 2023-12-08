//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception



// CONFIG C

#include <stdio.h>
#include <Block_private.h>


int main(int argc, char *argv[]) {
    void (^inner)(void) = ^ { printf("argc was %d\n", argc); };
    void (^outer)(void) = ^{
          inner();
          inner();
     };
     //printf("size of inner is %ld\n", Block_size(inner));
     //printf("size of outer is %ld\n", Block_size(outer));
     if (Block_size(inner) != Block_size(outer)) {
        printf("not the same size, using old compiler??\n");
        return 1;
    }
    printf("%s: Success\n", argv[0]);
    return 0;
}
