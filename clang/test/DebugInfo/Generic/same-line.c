// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -debug-info-kind=limited -o - | FileCheck %s
// Here two temporary nodes are identical (but should not get uniqued) while
// building the full debug type.
typedef struct { long x; } foo; typedef struct {  foo *x; } bar;
// CHECK: !DICompositeType(tag: DW_TAG_structure_type,{{.*}} line: 4, size: 64,
// CHECK: !DICompositeType(tag: DW_TAG_structure_type,{{.*}} line: 4, size: 64,
bar b;
