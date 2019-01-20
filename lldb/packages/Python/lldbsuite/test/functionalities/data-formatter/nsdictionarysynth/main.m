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


	NSArray* keys = @[@"foo",@"bar",@"baz"];
	NSArray* values = @[@"hello",@[@"X",@"Y"],@{@1 : @"one",@2 : @"two"}];
	NSDictionary* dictionary = [NSDictionary dictionaryWithObjects:values forKeys:keys];
	NSMutableDictionary* mutabledict = [NSMutableDictionary dictionaryWithCapacity:5];
	[mutabledict setObject:@"123" forKey:@23];
	[mutabledict setObject:[NSURL URLWithString:@"http://www.apple.com"] forKey:@"foobar"];
	[mutabledict setObject:@[@"a",@12] forKey:@57];
	[mutabledict setObject:dictionary forKey:@"sourceofstuff"];

    [pool drain];// Set break point at this line.
    return 0;
}

