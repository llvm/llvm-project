// RUN: %clang_cc1 %s -triple=i686-pc-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -o - | FileCheck %s

// Tests virtual bases in the MS ABI.

// CHECK: ![[NoPrimaryBase:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "NoPrimaryBase",
// CHECK-SAME: elements: ![[elements:[0-9]+]]

// CHECK: ![[elements]] = !{![[NoPrimaryBase_base:[0-9]+]]}

// CHECK: ![[NoPrimaryBase_base]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[NoPrimaryBase]],
// CHECK-SAME: baseType: ![[HasVirtualMethod:[0-9]+]], offset: 4, flags: DIFlagVirtual, extraData: i32 0)

// CHECK: ![[HasVirtualMethod]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "HasVirtualMethod"

// CHECK: ![[HasPrimaryBase:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "HasPrimaryBase",
// CHECK-SAME: elements: ![[elements:[0-9]+]]

// CHECK: ![[elements]] = !{![[SecondaryVTable_base:[0-9]+]], ![[HasVirtualMethod_base:[0-9]+]], ![[vshape:[0-9]+]]}

// CHECK: ![[SecondaryVTable_base]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[HasPrimaryBase]],
// CHECK-SAME: baseType: ![[SecondaryVTable:[0-9]+]], offset: 4, flags: DIFlagVirtual, extraData: i32 4)

// CHECK: ![[SecondaryVTable]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "SecondaryVTable"

// CHECK: ![[HasVirtualMethod_base]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[HasPrimaryBase]], baseType: ![[HasVirtualMethod]], extraData: i32 0)

// CHECK: ![[HasIndirectVirtualBase:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "HasIndirectVirtualBase"
// CHECK-SAME: elements: ![[elements:[0-9]+]]

// CHECK: !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[HasIndirectVirtualBase]], baseType: ![[HasPrimaryBase]]
// CHECK-NOT: DIFlagIndirectVirtualBase
// CHECK-SAME: )

// CHECK: !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[HasIndirectVirtualBase]], baseType: ![[SecondaryVTable]]
// CHECK-SAME: flags:
// CHECK-SAME: DIFlagIndirectVirtualBase

// CHECK: ![[DynamicNoVFPtr:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "DynamicNoVFPtr",
// CHECK-SAME: elements: ![[elements:[0-9]+]]

// CHECK: ![[elements]] = !{![[POD_base:[0-9]+]]}

// CHECK: ![[POD_base]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[DynamicNoVFPtr]],
// CHECK-SAME: baseType: ![[POD:[0-9]+]], offset: 4, flags: DIFlagVirtual, extraData: i32 0)

// CHECK: ![[POD]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "POD"

struct POD { int pod; };

struct DynamicNoVFPtr : virtual POD { };

DynamicNoVFPtr dynamic_no_vfptr;

struct HasVirtualMethod { virtual void f(); };

struct NoPrimaryBase : virtual HasVirtualMethod { };

NoPrimaryBase no_primary_base;

struct SecondaryVTable { virtual void g(); };

struct HasPrimaryBase : virtual SecondaryVTable, HasVirtualMethod { };

HasPrimaryBase has_primary_base;

struct HasIndirectVirtualBase : public HasPrimaryBase {};

HasIndirectVirtualBase has_indirect_virtual_base;
