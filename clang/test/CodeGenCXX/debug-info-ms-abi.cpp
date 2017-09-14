// RUN: %clang_cc1 %s -triple=i686-pc-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -o - | FileCheck %s

// Tests that certain miscellaneous features work in the MS ABI.

struct Foo {
  virtual void f();
  virtual void g();
  virtual void h();
  static void i(int, int);
  struct Nested {};
};
Foo f;
Foo::Nested n;

// CHECK: ![[Nested:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nested",
// CHECK-SAME: identifier: ".?AUNested@Foo@@"

// CHECK: ![[Foo:[^ ]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo",
// CHECK-SAME: elements: ![[elements:[0-9]+]]
// CHECK-SAME: identifier: ".?AUFoo@@"

// CHECK: ![[elements]] = !{![[vshape:[0-9]+]], ![[vptr:[0-9]+]], ![[Nested]], ![[f:[0-9]+]], ![[g:[0-9]+]], ![[h:[0-9]+]], ![[i:[0-9]+]]}

// CHECK: ![[vshape]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: null, size: 96)

// CHECK: ![[vptr]] = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$Foo",
// CHECK-SAME: baseType: ![[vptr_ty:[0-9]+]],

// CHECK: ![[vptr_ty]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[vshape]], size: 32

// CHECK: ![[f]] = !DISubprogram(name: "f",
// CHECK-SAME: containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0,
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagIntroducedVirtual,

// CHECK: ![[g]] = !DISubprogram(name: "g",
// CHECK-SAME: containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1,
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagIntroducedVirtual,

// CHECK: ![[h]] = !DISubprogram(name: "h",
// CHECK-SAME: containingType: ![[Foo]], virtuality: DW_VIRTUALITY_virtual, virtualIndex: 2,
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagIntroducedVirtual,

// CHECK: ![[i]] = !DISubprogram(name: "i",
// CHECK-SAME: flags: DIFlagPrototyped | DIFlagStaticMember
// CHECK-NEXT: ![[dummy:[0-9]+]] = !DISubroutineType(types: ![[Signature:[0-9]+]])
// CHECK: ![[Signature]] = !{null, ![[BasicInt:[0-9]+]], ![[BasicInt]]}
// CHECK: ![[BasicInt]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
