// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-linux-gnu  %s -o - | FileCheck %s

// Make sure that the union type has template parameters.

namespace PR15637 {
  template <typename T> union Value { int a; };
  void g(float value) {
    Value<float> tempValue;
  }
  Value<float> f;
}

// CHECK: !DICompositeType(tag: DW_TAG_union_type, name: "Value<float>",
// CHECK-SAME:             templateParams: [[TTPARAM:![0-9]+]]
// CHECK-SAME:             identifier: "_ZTSN7PR156375ValueIfEE"
// CHECK: [[TTPARAM]] = !{[[PARAMS:.*]]}
// CHECK: [[PARAMS]] = !DITemplateTypeParameter(name: "T"
