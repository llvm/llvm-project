//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <Block.h>

// CONFIG

void callsomething(const char *format, int argument) {
}

void
dispatch_call_Block_with_release2(void *block)
{
        void (^b)(void) = (void (^)(void))block;
        b();
        Block_release(b);
}

int main(int argc, char *argv[]) {
     void (^b1)(void) = ^{ callsomething("argc is %d\n", argc); };
     void (^b2)(void) = ^{ callsomething("hellow world\n", 0); }; // global block now

     dispatch_call_Block_with_release2(Block_copy(b1));
     dispatch_call_Block_with_release2(Block_copy(b2));
     printf("%s: Success\n", argv[0]);
     return 0;
}
