// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// Both properties should be emitted as having a class and an instance property
// with the same name is allowed.
@interface I1
// CHECK: !DIObjCProperty(name: "p1"
// CHECK-SAME:            line: [[@LINE+1]]
@property int p1;
// CHECK: !DIObjCProperty(name: "p1"
// CHECK-SAME:            line: [[@LINE+1]]
@property(class) int p1;
@end

@implementation I1
@synthesize p1;
@end

void foo(I1 *iptr) {}
