//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <CoreMedia/CoreMedia.h>

int main(int argc, const char **argv)
{
    @autoreleasepool
    {
        CMTime t1 = CMTimeMake(1, 10);
        CMTime t2 = CMTimeMake(10, 1);

        CMTimeShow(t1); // break here
        CMTimeShow(t2);
    }
    return 0;
}
