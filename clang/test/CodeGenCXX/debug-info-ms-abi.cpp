// RUN: %clang_cc1 %s -triple=i686-pc-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -o - | FileCheck %s

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
// CHECK-SAME: elements: ![[elements:[0-9]+]]
// CHECK-SAME: identifier: ".?AUFoo@@"

// CHECK: ![[elements]] = !{![[vptr:[0-9]+]], ![[Nested:[0-9]+]], ![[f:[0-9]+]], ![[g:[0-9]+]], ![[h:[0-9]+]]}

// CHECK: ![[Nested]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nested",
// CHECK-SAME: identifier: ".?AUNested@Foo@@"

// CHECK: ![[f]] = !DISubprogram(name: "f",
// CHECK-SAME: containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0,
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagIntroducedVirtual,

// CHECK: ![[g]] = !DISubprogram(name: "g",
// CHECK-SAME: containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1,
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagIntroducedVirtual,

// CHECK: ![[h]] = !DISubprogram(name: "h",
// CHECK-SAME: containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 2,
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagIntroducedVirtual,
