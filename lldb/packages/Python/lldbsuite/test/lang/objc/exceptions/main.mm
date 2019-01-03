//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

#import <exception>
#import <stdexcept>

@interface MyCustomException: NSException
@end
@implementation MyCustomException
@end

void foo(int n)
{
    NSDictionary *info = [NSDictionary dictionaryWithObjectsAndKeys:@"some_value", @"some_key", nil];
    switch (n) {
        case 0:
            @throw [[NSException alloc] initWithName:@"ThrownException" reason:@"SomeReason" userInfo:info];
        case 1:
            @throw [[MyCustomException alloc] initWithName:@"ThrownException" reason:@"SomeReason" userInfo:info];
        case 2:
            throw std::runtime_error("C++ exception");
    }
}

void rethrow(int n)
{
    @try {
        foo(n);
    } @catch(NSException *e) {
        @throw;
    }
}

int main(int argc, const char * argv[])
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSDictionary *info = [NSDictionary dictionaryWithObjectsAndKeys:@"some_value", @"some_key", nil];
    NSException *e1 = [[NSException alloc] initWithName:@"ExceptionName" reason:@"SomeReason" userInfo:info];
    NSException *e2;

    @try {
        foo(atoi(argv[1]));
    } @catch(NSException *e) {
        e2 = e;
    }

    NSLog(@"1"); // Set break point at this line.

    rethrow(atoi(argv[1]));

    [pool drain];
    return 0;
}

