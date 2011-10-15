//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

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
