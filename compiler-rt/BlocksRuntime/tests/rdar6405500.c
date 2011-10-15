//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// CONFIG rdar://6405500

#include <stdio.h>
#include <stdlib.h>
#import <dispatch/dispatch.h>
#import <objc/objc-auto.h>

int main (int argc, const char * argv[]) {
    __block void (^blockFu)(size_t t);
    blockFu = ^(size_t t){
        if (t == 20) {
            printf("%s: success\n", argv[0]);
            exit(0);
        } else
            dispatch_async(dispatch_get_main_queue(), ^{ blockFu(20); });
    };
    
    dispatch_apply(10, dispatch_get_concurrent_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT), blockFu);

    dispatch_main();
    printf("shouldn't get here\n");
    return 1;
}
