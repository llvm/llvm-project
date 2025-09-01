// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -debug-info-kind=constructor %s -o - | FileCheck %s

// Test that clang doesn't crash while resolving temporary debug metadata of
// a record with collisions in the record's enum users.

// CHECK:      !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-SAME:                  scope: [[SCOPE:![0-9]+]]
// CHECK-SAME:                  elements: [[ELEMENTS:![0-9]+]]
// CHECK:      [[SCOPE]] = !DICompositeType(tag: DW_TAG_structure_type
// CHECK-SAME:                              name: "Struct1<Struct3>"
// CHECK:      [[ELEMENTS]] = !{[[ELEMENT:![0-9]+]]}
// CHECK:      [[ELEMENT]] = !DIEnumerator(name: "enumValue1"

template <typename> struct Struct1 {
  enum { enumValue1 };
  Struct1();
};
void function2() {
  struct Struct3 {};
  int i = Struct1<Struct3>::enumValue1;
}
void function3() {
  struct Struct3 {};
  int i = Struct1<Struct3>::enumValue1;
}
