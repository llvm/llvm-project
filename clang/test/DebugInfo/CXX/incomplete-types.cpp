// RUN: %clang_cc1 -debug-info-kind=limited -gomit-unreferenced-methods %s -emit-llvm -o - | FileCheck %s

struct t1 {
  void f1();
  void f2();
};

void t1::f1() { }

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1"
// CHECK-SAME: elements: [[ELEMENTS:![0-9]+]]
// CHECK: [[ELEMENTS]] = !{}
