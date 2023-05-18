// RUN: not %clang  -fsyntax-only -fno-objc-default-synthesize-properties -fobjc-default-synthesize-properties %s 2>&1 | FileCheck %s

@interface I
@property int P;
@end

@implementation I
@end
// CHECK: error: unknown argument: '-fno-objc-default-synthesize-properties'
// CHECK: error: unknown argument: '-fobjc-default-synthesize-properties'
