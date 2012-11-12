// RUN: %clang -fverbose-asm -g -S -emit-llvm %s -o - | FileCheck %s
@class ForwardObjcClass;
ForwardObjcClass *ptr = 0;

// CHECK: metadata !{i32 {{.*}}, null, metadata !"ForwardObjcClass", metadata !{{.*}}, i32 2, i64 0, i64 0, i32 0, i32 4, null, null, i32 16} ; [ DW_TAG_structure_type ]
