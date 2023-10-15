// RUN: %clang -target x86_64-linux -g -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -verify -DMISMATCH %s

struct A {
  enum E : unsigned {};
  [[clang::debug_info_type(E)]] unsigned b : 2;
#ifdef MISMATCH
  [[clang::debug_info_type(E)]] int b2 : 2;
  // expected-warning@-1 {{underlying type 'unsigned int' of enumeration 'E' doesn't match bitfield type 'int'}}
#endif
} a;

// CHECK-DAG: [[ENUM:![0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E"{{.*}}
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "b",{{.*}} baseType: [[ENUM]]