//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

@interface MyClass : NSObject {
};
-(void)test;
@end

@implementation MyClass
-(void)test {
    printf("%p\n", self); // break here
}
@end

@interface MyOwner : NSObject {
  @public id ownedThing; // should be id, to test <rdar://problem/31363513>
};
@end

@implementation MyOwner
@end

int main (int argc, char const *argv[]) {
    @autoreleasepool {
        MyOwner *owner = [[MyOwner alloc] init];
        owner->ownedThing = [[MyClass alloc] init];
        [(MyClass*)owner->ownedThing test];
    }
    return 0;
}

