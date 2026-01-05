// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: ![[BASE_PROP:[0-9]+]] = !DIObjCProperty(name: "base"
// CHECK-SAME:                                    attributes: 2316
// CHECK-SAME:                                    type: ![[P1_TYPE:[0-9]+]]
//
// CHECK: ![[P1_TYPE]] = !DIBasicType(name: "int"
//
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "_customIvar"
// CHECK-SAME:           extraData: ![[BASE_PROP]]

@interface C {
  int _customIvar;
}
@property int base;
@end

@implementation C
@synthesize base = _customIvar;
@end

void foo(C *cptr) {}
