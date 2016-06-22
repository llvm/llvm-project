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

// CHECK: !DISubprogram(name: "f",
// CHECK-SAME: containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0,
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagIntroducedVirtual,

// CHECK: !DISubprogram(name: "g",
// CHECK-SAME: containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1,
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagIntroducedVirtual,

// CHECK: !DISubprogram(name: "h",
// CHECK-SAME: containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 2,
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagIntroducedVirtual,

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nested",
// CHECK-SAME: identifier: ".?AUNested@Foo@@"
