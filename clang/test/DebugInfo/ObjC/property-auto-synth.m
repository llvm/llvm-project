// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK-NOT: setter
// CHECK-NOT: getter

@interface I1
@property int p1;
@end

@implementation I1
@end

void foo(I1 *ptr) {}
