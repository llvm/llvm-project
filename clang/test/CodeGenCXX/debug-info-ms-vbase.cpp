// RUN: %clang_cc1 %s -triple=i686-pc-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -o - | FileCheck %s

// Tests virtual bases in the MS ABI.

struct POD { int pod; };

struct DynamicNoVFPtr : virtual POD { };

DynamicNoVFPtr dynamic_no_vfptr;

// CHECK: ![[DynamicNoVFPtr:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "DynamicNoVFPtr",
// CHECK-SAME: elements: ![[elements:[0-9]+]]

// CHECK: ![[elements]] = !{![[POD_base:[0-9]+]]}

// CHECK: ![[POD_base]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[DynamicNoVFPtr]],
// CHECK-SAME: baseType: ![[POD:[0-9]+]], offset: 4, flags: DIFlagVirtual)

// CHECK: ![[POD]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "POD"

struct HasVirtualMethod { virtual void f(); };

struct NoPrimaryBase : virtual HasVirtualMethod { };

NoPrimaryBase no_primary_base;

// CHECK: ![[NoPrimaryBase:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "NoPrimaryBase",
// CHECK-SAME: elements: ![[elements:[0-9]+]]

// CHECK: ![[elements]] = !{![[NoPrimaryBase_base:[0-9]+]]}

// CHECK: ![[NoPrimaryBase_base]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[NoPrimaryBase]],
// CHECK-SAME: baseType: ![[HasVirtualMethod:[0-9]+]], offset: 4, flags: DIFlagVirtual)

// CHECK: ![[HasVirtualMethod]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "HasVirtualMethod"

struct SecondaryVTable { virtual void g(); };

struct HasPrimaryBase : virtual SecondaryVTable, HasVirtualMethod { };

HasPrimaryBase has_primary_base;

// CHECK: ![[HasPrimaryBase:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "HasPrimaryBase",
// CHECK-SAME: elements: ![[elements:[0-9]+]]

// CHECK: ![[elements]] = !{![[SecondaryVTable_base:[0-9]+]], ![[HasVirtualMethod_base:[0-9]+]], ![[vshape:[0-9]+]]}

// CHECK: ![[SecondaryVTable_base]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[HasPrimaryBase]],
// CHECK-SAME: baseType: ![[SecondaryVTable:[0-9]+]], offset: 4, flags: DIFlagVirtual)

// CHECK: ![[SecondaryVTable]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "SecondaryVTable"

// CHECK: ![[HasVirtualMethod_base]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[HasPrimaryBase]], baseType: ![[HasVirtualMethod]])

