// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: @gGlobals = external global

@interface I
- (int) Meth;
@end

@implementation I
- (int) Meth {
    extern int gGlobals;
    return gGlobals;
}
@end
