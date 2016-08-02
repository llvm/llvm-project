// Test that debug info is emitted for an Objective-C module and
// a precompiled header.

// REQUIRES: asserts, shell

// Modules:
// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules -fmodule-format=obj \
// RUN:   -fimplicit-module-maps -DMODULES -fmodules-cache-path=%t %s \
// RUN:   -I %S/Inputs -I %t -emit-llvm -o %t.ll \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-mod.ll
// RUN: cat %t-mod.ll | FileCheck %s
// RUN: cat %t-mod.ll | FileCheck %s --check-prefix=MODULE-CHECK

// PCH:
// RUN: %clang_cc1 -x objective-c -emit-pch -fmodule-format=obj -I %S/Inputs \
// RUN:   -o %t.pch %S/Inputs/DebugObjC.h \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-pch.ll
// RUN: cat %t-pch.ll | FileCheck %s

#ifdef MODULES
@import DebugObjC;
#endif

// CHECK: distinct !DICompileUnit(language: DW_LANG_ObjC
// CHECK-SAME:                    isOptimized: false,
// CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:             name: "FwdDecl",
// CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:             name: "ObjCClass",
// CHECK: !DIObjCProperty(name: "property",
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "ivar"
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "InnerEnum"
// CHECK: !DISubprogram(name: "+[ObjCClass classMethod]"
// CHECK: !DISubprogram(name: "-[ObjCClass instanceMethodWithInt:]"
// CHECK: !DISubprogram(name: "-[Category(Category) categoryMethod]"

// MODULE-CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// MODULE-CHECK-SAME:             scope: ![[MODULE:[0-9]+]],
// MODULE-CHECK: ![[MODULE]] = !DIModule(scope: null, name: "DebugObjC"
// MODULE-CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// MODULE-CHECK-SAME:             name: "FwdDecl",
// MODULE-CHECK-SAME:             scope: ![[MODULE]],
// MODULE-CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// MODULE-CHECK-SAME:             name: "ObjCClass",
// MODULE-CHECK-SAME:             scope: ![[MODULE]],
// MODULE-CHECK: !DISubprogram(name: "+[ObjCClass classMethod]",
// MODULE-CHECK-SAME:          scope: ![[MODULE]],

// The forward declaration should not be in the module scope.
// MODULE-CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "OpaqueData", file
