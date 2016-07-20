//===-- main.m -------------------------------------------*- Objective-C-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

@interface VarClass : NSObject
- (id) lottaArgs: (id) first, ...;
@end

@implementation VarClass
- (id) lottaArgs: (id) first, ...
{
  return first;
}
@end

int
main()
{
  VarClass *my_var = [[VarClass alloc] init];
  id something = [my_var lottaArgs: @"111", @"222", nil];
  NSLog (@"%@ - %@", my_var, something); //% self.expect("expression -O -- [my_var lottaArgs:@\"111\", @\"222\", nil]", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["111"])
  return 0;
}

