//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

void foo()
{
    NSDictionary *info = [NSDictionary dictionaryWithObjectsAndKeys:@"some_value", @"some_key", nil];
    @throw [[NSException alloc] initWithName:@"ThrownException" reason:@"SomeReason" userInfo:info];
}

int main (int argc, const char * argv[])
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSDictionary *info = [NSDictionary dictionaryWithObjectsAndKeys:@"some_value", @"some_key", nil];
    NSException *e1 = [[NSException alloc] initWithName:@"ExceptionName" reason:@"SomeReason" userInfo:info];
    NSException *e2;

    @try {
        foo();
    } @catch(NSException *e) {
        e2 = e;
    }

    NSLog(@"1"); // Set break point at this line.
    [pool drain];
    return 0;
}

