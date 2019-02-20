//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

int main (int argc, const char * argv[])
{
    
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	NSSet* set = [NSSet setWithArray:@[@1,@"hello",@2,@"world"]];
	NSMutableSet* mutable = [NSMutableSet setWithCapacity:5];
	[mutable addObject:@1];
	[mutable addObject:@2];
	[mutable addObject:@3];
	[mutable addObject:@4];
	[mutable addObject:@5];
	[mutable addObject:[NSURL URLWithString:@"www.apple.com"]];
	[mutable addObject:@[@1,@2,@3]];
	[mutable unionSet:set];
	[mutable removeAllObjects]; // Set break point at this line.
	[mutable unionSet:set];
	[mutable addObject:@1];

    [pool drain];
    return 0;
}

