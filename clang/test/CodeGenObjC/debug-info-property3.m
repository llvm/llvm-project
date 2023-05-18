// RUN: %clang_cc1 -S -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

@interface I1
// CHECK: !DIObjCProperty(name: "p1"
// CHECK-SAME:            line: [[@LINE+2]]
// CHECK-SAME:            attributes: 2316
@property int p1;
@end

@implementation I1
@synthesize p1;
@end

void foo(I1 *iptr) {}
