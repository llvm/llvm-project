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
#import <stdarg.h>


int main (int argc, const char * argv[]) {
    int (^sumn)(int n, ...) = ^(int n, ...){
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
    int six = sumn(3, 1, 2, 3);
    
    if ( six != 6 ) {
        printf("%s: Expected 6 but got %d\n", argv[0], six);
        exit(1);
    }
    
    printf("%s: success\n", argv[0]);
    return 0;
}
