// RUN: %clang -S -emit-llvm %s -o - -x objective-c -fobjc-runtime=gnustep-1.5 | FileCheck  %s

// Regression test: check that we don't crash when referencing a forward-declared protocol.
@protocol P;

@interface I <P>
@end

@implementation I

@end

// CHECK: @.objc_protocol
