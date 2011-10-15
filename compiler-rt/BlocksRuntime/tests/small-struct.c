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
  int a;
  int b;
} MiniStruct;

int main (int argc, const char * argv[]) {
    MiniStruct inny;
    MiniStruct outty;
    MiniStruct (^copyStruct)(MiniStruct);
    
    memset(&inny, 0xA5, sizeof(inny));
    memset(&outty, 0x2A, sizeof(outty));    
    
    inny.a = 12;
    inny.b = 42;

    copyStruct = ^(MiniStruct aTinyStruct){ return aTinyStruct; };  // pass-by-value intrinsically copies the argument
    
    outty = copyStruct(inny);

    if ( &inny == &outty ) {
        printf("%s: struct wasn't copied.", argv[0]);
        exit(1);
    }
    if ( (inny.a != outty.a) || (inny.b != outty.b) ) {
        printf("%s: struct contents did not match.", argv[0]);
        exit(1);
    }
    
    printf("%s: success\n", argv[0]);
    return 0;
}
