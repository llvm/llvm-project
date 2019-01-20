//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *  variadic.c
 *  testObjects
 *
 *  Created by Blaine Garst on 2/17/09.
 *
 */

// PURPOSE Test that variadic arguments compile and work for Blocks
// CONFIG

#include <stdarg.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    
    long (^addthem)(const char *, ...) = ^long (const char *format, ...){
        va_list argp;
        const char *p;
        int i;
        char c;
        double d;
        long result = 0;
        va_start(argp, format);
        //printf("starting...\n");
        for (p = format; *p; p++) switch (*p) {
            case 'i':
                i = va_arg(argp, int);
                //printf("i: %d\n", i);
                result += i;
                break;
            case 'd':
                d = va_arg(argp, double);
                //printf("d: %g\n", d);
                result += (int)d;
                break;
            case 'c':
                c = va_arg(argp, int);
                //printf("c: '%c'\n", c);
                result += c;
                break;
        }
        //printf("...done\n\n");
        return result;
    };
    long testresult = addthem("ii", 10, 20);
    if (testresult != 30) {
        printf("got wrong result: %ld\n", testresult);
        return 1;
    }
    testresult = addthem("idc", 30, 40.0, 'a');
    if (testresult != (70+'a')) {
        printf("got different wrong result: %ld\n", testresult);
        return 1;
    }
    printf("%s: Success\n", argv[0]);
    return 0;
}


