// RUN: %clang -target x86_64-linux -g -S -emit-llvm -o - %s | FileCheck %s

struct A {
  enum E : unsigned {};
  [[clang::preferred_type(E)]] unsigned b : 2;
} a;

// CHECK-DAG: [[ENUM:![0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E"{{.*}}
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "b",{{.*}} baseType: [[ENUM]]
