//  Test for debug info related to DW_AT_alignment attribute in the struct type.
// RUN: %clang_cc1 -dwarf-version=5 -debug-info-kind=standalone -S -emit-llvm %s -o - | FileCheck %s

// CHECK-DAG: DICompositeType(tag: DW_TAG_structure_type, name: "MyType", {{.*}}, align: 32
// CHECK-DAG: DICompositeType(tag: DW_TAG_structure_type, name: "MyType1", {{.*}}, align: 8
// CHECK-DAG: DICompositeType(tag: DW_TAG_structure_type, name: "MyType2", {{.*}}, align: 8

struct MyType {
  int m;
} __attribute__((aligned(1)));
MyType mt;

static_assert(alignof(MyType) == 4, "alignof MyType is wrong");

struct MyType1 {
  int m;
} __attribute__((packed, aligned(1)));
MyType1 mt1;

static_assert(alignof(MyType1) == 1, "alignof MyType1 is wrong");

struct MyType2 {
  __attribute__((packed)) int m;
} __attribute__((aligned(1)));
MyType2 mt2;

static_assert(alignof(MyType2) == 1, "alignof MyType2 is wrong");
