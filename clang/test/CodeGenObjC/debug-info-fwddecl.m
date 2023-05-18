// RUN: %clang -fverbose-asm -g -S -emit-llvm %s -o - | FileCheck %s
@class ForwardObjcClass;
ForwardObjcClass *ptr = 0;

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "ForwardObjcClass"
// CHECK-SAME:             line: 2
// CHECK-NOT:              size:
// CHECK-NOT:              align:
// CHECK-NOT:              offset:
// CHECK-SAME:             flags: DIFlagFwdDecl
