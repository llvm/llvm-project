// RUN: %clang  -fsyntax-only -fno-objc-default-synthesize-properties -fobjc-default-synthesize-properties %s 2>&1 | FileCheck %s

@interface I
@property int P;
@end

@implementation I
@end
// CHECK: warning: argument unused during compilation: '-fno-objc-default-synthesize-properties'
// CHECK: warning: argument unused during compilation: '-fobjc-default-synthesize-properties'
