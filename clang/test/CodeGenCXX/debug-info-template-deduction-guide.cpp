// RUN: %clang -S -emit-llvm -target x86_64-unknown_unknown -g %s -o - -std=c++1z | FileCheck %s

// Verify that we don't crash when emitting debug information for objects
// created from a deduced template specialization.

template <class T>
struct S {
  S(T) {}
};

// CHECK: !DIGlobalVariable(name: "s1"
// CHECK-SAME: type: [[TYPE_NUM:![0-9]+]]
// CHECK: !DIGlobalVariable(name: "s2"
// CHECK-SAME: type: [[TYPE_NUM]]
// CHECK: [[TYPE_NUM]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S<int>",
S s1(42);
S<int> s2(42);
