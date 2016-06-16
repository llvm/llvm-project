// RUN: %clang_cc1 %s -triple=i686-pc-windows-msvc -debug-info-kind=limited -emit-llvm -o - | FileCheck %s

// Tests that certain miscellaneous features work in the MS ABI.

struct Foo {
  struct Nested {};
};
Foo f;
Foo::Nested n;
// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo",
// CHECK-SAME: identifier: ".?AUFoo@@"
// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nested",
// CHECK-SAME: identifier: ".?AUNested@Foo@@"
