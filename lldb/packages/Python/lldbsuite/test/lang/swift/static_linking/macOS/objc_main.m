//===-- objc_main.m ---------------------------------------------*- C++ -*-===//
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

#import "A-Swift.h"
#import "B-Swift.h"

int main(int argc, const char * argv[]) {
  @autoreleasepool {
      NSLog(@"Hello, World!");
      NSLog(@"A = %ld", [[[A alloc] init] foo]);
      NSLog(@"B = %ld", [[[B alloc] init] bar]);
  }
	return 0;
}
