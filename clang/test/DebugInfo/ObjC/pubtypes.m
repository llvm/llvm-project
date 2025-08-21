// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "H"
// CHECK-SAME:             line: [[@LINE+1]],
@interface H
-(void) foo;
@end

@implementation H
-(void) foo {
}
@end

