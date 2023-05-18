//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CONFIG open rdar://6439600

#import <stdio.h>
#import <stdlib.h>

#define NUMBER_OF_BLOCKS 100
int main (int argc, const char * argv[]) {
    int (^x[NUMBER_OF_BLOCKS])();
    int i;
    
    for(i=0; i<NUMBER_OF_BLOCKS; i++) x[i] = ^{ return i; };

    for(i=0; i<NUMBER_OF_BLOCKS; i++) {
        if (x[i]() != i) {
            printf("%s: failure, %d != %d\n", argv[0], x[i](), i);
            exit(1);
        }
    }
    
    printf("%s: success\n", argv[0]);
    
    return 0;
}
