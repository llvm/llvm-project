//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

//  -*- mode:C; c-basic-offset:4; tab-width:4; intent-tabs-mode:nil;  -*-
// CONFIG

#import <stdio.h>
#import <stdlib.h>
#import <string.h>

typedef struct {
    unsigned long ps[30];
    int qs[30];
} BobTheStruct;

int main (int argc, const char * argv[]) {
    BobTheStruct inny;
    BobTheStruct outty;
    BobTheStruct (^copyStruct)(BobTheStruct);
    int i;
    
    memset(&inny, 0xA5, sizeof(inny));
    memset(&outty, 0x2A, sizeof(outty));    
    
    for(i=0; i<30; i++) {
        inny.ps[i] = i * i * i;
        inny.qs[i] = -i * i * i;
    }
    
    copyStruct = ^(BobTheStruct aBigStruct){ return aBigStruct; };  // pass-by-value intrinsically copies the argument
    
    outty = copyStruct(inny);

    if ( &inny == &outty ) {
        printf("%s: struct wasn't copied.", argv[0]);
        exit(1);
    }
    for(i=0; i<30; i++) {
        if ( (inny.ps[i] != outty.ps[i]) || (inny.qs[i] != outty.qs[i]) ) {
            printf("%s: struct contents did not match.", argv[0]);
            exit(1);
        }
    }
    
    printf("%s: success\n", argv[0]);
    
    return 0;
}
