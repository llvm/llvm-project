// Test that debug info is emitted for an Objective-C module and
// a precompiled header.

// REQUIRES: asserts

// Modules:
// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules -fmodule-format=obj \
// RUN:   -fimplicit-module-maps -DMODULES -fmodules-cache-path=%t %s \
// RUN:   -I %S/Inputs -I %t -emit-llvm -o %t.ll \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-mod.ll
// RUN: cat %t-mod.ll | FileCheck %s
// RUN: cat %t-mod.ll | FileCheck %s --check-prefix=CHECK2

// PCH:
// RUN: %clang_cc1 -x objective-c -emit-pch -fmodule-format=obj -I %S/Inputs \
// RUN:   -o %t.pch %S/Inputs/DebugObjC.h \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-pch.ll
// RUN: cat %t-pch.ll | FileCheck %s
// RUN: cat %t-pch.ll | FileCheck %s --check-prefix=CHECK2

#ifdef MODULES
@import DebugObjC;
#endif

// CHECK: distinct !DICompileUnit(language: DW_LANG_ObjC, file: ![[FILE:[0-9]+]],
// CHECK-SAME:                    isOptimized: false

// CHECK: ![[FILE]] = !DIFile(filename: "{{DebugObjC|.*DebugObjC.h}}"

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-SAME:             scope: ![[MODULE:[0-9]+]],
// CHECK: ![[MODULE]] = !DIModule(scope: null, name: "DebugObjC

// CHECK: ![[TD_ENUM:.*]] = !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-NOT:              name:
// CHECK-SAME:             elements:

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "FwdDecl",
// CHECK-SAME:             scope: ![[MODULE]],

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "ObjCClass",
// CHECK-SAME:             scope: ![[MODULE]],
// CHECK-SAME:             elements

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "ObjCClassWithPrivateIVars",
// CHECK-SAME:             scope: ![[MODULE]],
// CHECK-SAME:             elements

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "FwdDeclared"
// CHECK-SAME:             elements:

// CHECK: ![[TD_UNION:.*]] = distinct !DICompositeType(tag: DW_TAG_union_type,
// CHECK-NOT:              name:
// CHECK-SAME:             elements:

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "TypedefUnion",
// CHECK-SAME:           baseType: ![[TD_UNION]])

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "TypedefEnum",
// CHECK-SAME:           baseType: ![[TD_ENUM:.*]])

// CHECK: ![[TD_STRUCT:.*]] = distinct !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-NOT:              name:
// CHECK-SAME:             elements:
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "TypedefStruct",
// CHECK-SAME:           baseType: ![[TD_STRUCT]])

// CHECK: !DICompositeType(tag: DW_TAG_union_type,
// CHECK-NOT:              name:
// CHECK-SAME:             )

// CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-NOT:              name:
// CHECK-SAME:             )

// CHECK-NEG-NOT: !DICompositeType(tag: DW_TAG_structure_type, name: "PureForwardDecl"

// The output order is sublty different for module vs. pch,
// so these are checked separately:
//
// CHECK2: !DICompositeType(tag: DW_TAG_structure_type, name: "FwdDecl",
// CHECK2: !DICompositeType(tag: DW_TAG_structure_type, name: "ObjCClass",
// CHECK2: !DIObjCProperty(name: "property",
// CHECK2: !DIDerivedType(tag: DW_TAG_member, name: "ivar"
// CHECK2: !DIDerivedType(tag: DW_TAG_typedef, name: "InnerEnum"
