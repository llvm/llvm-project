// Checks basic debug-info generation for property. Makes sure we
// create a DIObjCProperty for the synthesized property.

// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: !DIObjCProperty(name: "p1"
// CHECK-SAME:            attributes: 2316
// CHECK-SAME:            type: ![[P1_TYPE:[0-9]+]]
//
// CHECK: ![[P1_TYPE]] = !DIBasicType(name: "int"

@interface I1 {
int p1;
}
@property int p1;
@end

@implementation I1
@synthesize p1;
@end
