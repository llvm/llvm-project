// RUN: %clang_cc1 %s -std=c++14 -debug-info-kind=limited -emit-llvm -o - | FileCheck %s


// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B",
// CHECK-SAME: elements: ![[empty:[0-9]+]]
// CHECK: ![[empty]] = !{}

struct B {
  template <typename... e>
  static const int d = 0;
  template <typename e>
  static const auto d<e> = d<e, e>;
} c;
