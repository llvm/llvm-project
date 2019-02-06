//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
