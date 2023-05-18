//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <CoreFoundation/CoreFoundation.h>

#include <dispatch/dispatch.h>
#include <unistd.h>
//#import <Foundation/Foundation.h>
#include <Block.h>

// CONFIG rdar://problem/6371811

const char *whoami = "nobody";

void EnqueueStuff(dispatch_queue_t q)
{
    __block CFIndex counter;
    
    // above call has a side effect: it works around:
    // <rdar://problem/6225809> __block variables not implicitly imported into intermediate scopes
    dispatch_async(q, ^{
        counter = 0;
    });
    
    
    dispatch_async(q, ^{
        //printf("outer block.\n");
        counter++;
        dispatch_async(q, ^{
            //printf("inner block.\n");
            counter--;
            if(counter == 0) {
                printf("%s: success\n", whoami);
                exit(0);
            }
        });
        if(counter == 0) {
            printf("already done? inconceivable!\n");
            exit(1);
        }
    });        
}

int main (int argc, const char * argv[]) {
    dispatch_queue_t q = dispatch_queue_create("queue", NULL);

    whoami = argv[0];
    
    EnqueueStuff(q);
    
    dispatch_main();
    printf("shouldn't get here\n");
    return 1;
}
