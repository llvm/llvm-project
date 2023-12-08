//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//  -*- mode:C; c-basic-offset:4; tab-width:4; intent-tabs-mode:nil;  -*-
// HACK ALERT: gcc and g++ give different errors, referencing the line number to ensure that it checks for the right error; MUST KEEP IN SYNC WITH THE TEST
// CONFIG 27: error:

#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <stdarg.h>


int main (int argc, const char * argv[]) {
    int (^sumn)(int n, ...);
    int six = 0;
    
    sumn = ^(int a, int b, int n, ...){
        int result = 0;
        va_list numbers;
        int i;

        va_start(numbers, n);
        for (i = 0 ; i < n ; i++) {
            result += va_arg(numbers, int);
        }
        va_end(numbers);

        return result;
    };

    six = sumn(3, 1, 2, 3);

    if ( six != 6 ) {
        printf("%s: Expected 6 but got %d\n", argv[0], six);
        exit(1);
    }
    
    printf("%s: success\n", argv[0]);
    return 0;
}
