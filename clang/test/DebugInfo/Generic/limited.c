// RUN: %clang -flimit-debug-info -emit-llvm -g -S %s -o - | FileCheck %s

// Ensure we emit the full definition of 'foo' even though only its declaration
// is needed, since C has no ODR to ensure that the definition will be the same
// in whatever TU actually uses/requires the definition of 'foo'.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo",
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}

struct foo {
};

struct foo *f;
