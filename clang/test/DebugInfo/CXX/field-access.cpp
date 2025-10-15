// RUN: %clang -g -S -emit-llvm %s -o - | FileCheck %s

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "p"
// CHECK-SAME:           baseType: ![[INT:[0-9]+]]
// CHECK-SAME:           DIFlagPublic
// CHECK: ![[INT]] = !DIBasicType(name: "int"
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "pr"
// CHECK-NOT:            flags:
// CHECK-SAME:           baseType: ![[INT]]

class A {
public:
  int p;
private:
  int pr;
};

A a;
