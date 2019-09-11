//===-- MyClass.mm ----------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#import <Cocoa/Cocoa.h>
#import "MyClass.h"

class Base {
private:
    int m_BaseX = -1;
public:
    virtual ~Base () {}
};

class Derived: public Base {
private:
    int m_DerivedX;
public:
    Derived() : Base() {
        m_DerivedX = 1;
    }
    virtual ~Derived() {}
};

struct IntIntPair {
    int A;
    int B;
};

@implementation MyClass {
    IntIntPair m_pair;
    Base *m_base;
    NSArray *m_myclass_numbers;
}
- (id)init {
    if (self = [super init]) {
        self->m_pair.A = 1;
        self->m_pair.B = 2;
        self->m_base = new Derived();
        self->m_myclass_numbers = @[@1, @2, @3];
    }
    return self;
}
@end

@implementation MySubclass {
    NSString *m_mysubclass_s;
    NSRect m_mysubclass_r;
}
- (id)init {
    if (self = [super init]) {
        self->m_mysubclass_s = @"an NSString here";
        self->m_subclass_ivar = 42;
        self->m_mysubclass_r = NSMakeRect(0, 0, 30, 40);
    }
    return self;
}
@end

@implementation MySillyOtherClass {
    NSURL *url;
}
- (id)init {
    if (self = [super init]) {
        self->x = 12;
        self->url = [NSURL URLWithString:@"http://www.apple.com"];
    }
    return self;
}

@end
