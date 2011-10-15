//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

//
//  c99.m
//
// CONFIG C99 rdar://problem/6399225

#import <stdio.h>
#import <stdlib.h>

int main(int argc, char *argv[]) {
    void (^blockA)(void) = ^ { ; };
    blockA();
    printf("%s: success\n", argv[0]);
    exit(0);
}
