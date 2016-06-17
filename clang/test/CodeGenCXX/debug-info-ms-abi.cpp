// RUN: %clang_cc1 %s -triple=i686-pc-windows-msvc -debug-info-kind=limited -emit-llvm -o - | FileCheck %s

// Tests that certain miscellaneous features work in the MS ABI.

struct Foo {
  virtual void f();
  virtual void g();
  virtual void h();
  struct Nested {};
};
Foo f;
Foo::Nested n;
// CHECK: ![[Foo:[^ ]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo",
// CHECK-SAME: identifier: ".?AUFoo@@"
// CHECK: !DISubprogram(name: "f", {{.*}} containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, {{.*}})
// CHECK: !DISubprogram(name: "g", {{.*}} containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, {{.*}})
// CHECK: !DISubprogram(name: "h", {{.*}} containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 2, {{.*}})
// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nested",
// CHECK-SAME: identifier: ".?AUNested@Foo@@"
