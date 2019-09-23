//===-- MyClass.h -----------------------------------------------*- C++ -*-===//
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

#import <Foundation/Foundation.h>

@interface MyClass : NSObject
- (id)init;
@end

@interface MySubclass : MyClass {
    int m_subclass_ivar;
}
- (id)init;
@end

@interface MySillyOtherClass : NSObject {
    int x;
}
- (id)init;
@end
