//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.



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
