//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    __block BobTheStruct fiddly;
    BobTheStruct copy;

    void (^incrementFiddly)() = ^{
        int i;
        for(i=0; i<30; i++) {
            fiddly.ps[i]++;
            fiddly.qs[i]++;
        }
    };
    
    memset(&fiddly, 0xA5, sizeof(fiddly));
    memset(&copy, 0x2A, sizeof(copy));    
    
    int i;
    for(i=0; i<30; i++) {
        fiddly.ps[i] = i * i * i;
        fiddly.qs[i] = -i * i * i;
    }
    
    copy = fiddly;
    incrementFiddly();

    if ( &copy == &fiddly ) {
        printf("%s: struct wasn't copied.", argv[0]);
        exit(1);
    }
    for(i=0; i<30; i++) {
        //printf("[%d]: fiddly.ps: %lu, copy.ps: %lu, fiddly.qs: %d, copy.qs: %d\n", i, fiddly.ps[i], copy.ps[i], fiddly.qs[i], copy.qs[i]);
        if ( (fiddly.ps[i] != copy.ps[i] + 1) || (fiddly.qs[i] != copy.qs[i] + 1) ) {
            printf("%s: struct contents were not incremented.", argv[0]);
            exit(1);
        }
    }
    
    printf("%s: success\n", argv[0]);
    return 0;
}
