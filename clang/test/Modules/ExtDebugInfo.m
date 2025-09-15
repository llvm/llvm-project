// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// Test that only forward declarations are emitted for types defined in modules.

// Modules:
// RUN: %clang_cc1 -x objective-c -debug-info-kind=limited -dwarf-ext-refs -fmodules \
// RUN:     -fmodule-format=obj -fimplicit-module-maps -DMODULES \
// RUN:     -fmodules-cache-path=%t %s -I %S/Inputs -I %t -emit-llvm -o %t-mod.ll
// RUN: cat %t-mod.ll |  FileCheck %s
// RUN: cat %t-mod.ll |  FileCheck %s --check-prefix=DWOID

// PCH:
// RUN: %clang_cc1 -x objective-c -fmodule-format=obj -emit-pch -I%S/Inputs \
// RUN:     -o %t.pch %S/Inputs/DebugObjC.h
// RUN: %clang_cc1 -x objective-c -debug-info-kind=limited -dwarf-ext-refs \
// RUN:     -fmodule-format=obj \
// RUN:     -include-pch %t.pch %s -emit-llvm -o %t-pch.ll %s
// RUN: cat %t-pch.ll |  FileCheck %s
// RUN: cat %t-pch.ll |  FileCheck %s --check-prefix=DWOID

#ifdef MODULES
@import DebugObjC;
#endif

@implementation ObjCClassWithPrivateIVars {
  int hidden_ivar;
}
@end

TypedefUnion tdu;
TypedefEnum tde;
TypedefStruct tds;

int foo(ObjCClass *c) {
  InnerEnum e = e0;
  GlobalStruct.i = GlobalUnion.i = GlobalEnum;
  [c instanceMethodWithInt: 0];
  return [c property];
}

// DWOID: !DICompileUnit(language: DW_LANG_ObjC,{{.*}}isOptimized: false,{{.*}}dwoId:

// CHECK: ![[MOD:.*]] = !DIModule(scope: null, name: "DebugObjC

// CHECK: !DIGlobalVariable(name: "GlobalUnion",
// CHECK-SAME:              type: ![[GLOBAL_UNION:[0-9]+]]
// CHECK: ![[GLOBAL_UNION]] = distinct !DICompositeType(tag: DW_TAG_union_type,
// CHECK-SAME:                elements: !{{[0-9]+}})

// CHECK: !DIGlobalVariable(name: "GlobalStruct",
// CHECK-SAME:              type: ![[GLOBAL_STRUCT:[0-9]+]]
// CHECK: ![[GLOBAL_STRUCT]] = distinct !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:                elements: !{{[0-9]+}})

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "ObjCClassWithPrivateIVars",
// CHECK-SAME:             flags: DIFlagObjcClassComplete

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "hidden_ivar",
// CHECK-SAME:           flags: DIFlagPrivate)

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "TypedefEnum",
// CHECK-SAME:           baseType: ![[TD_ENUM:.*]])
// CHECK: ![[TD_ENUM]] = !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-SAME:             flags: DIFlagFwdDecl)

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "TypedefStruct",
// CHECK-SAME:           baseType: ![[TD_STRUCT:.*]])
// CHECK: ![[TD_STRUCT]] = !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:             flags: DIFlagFwdDecl)

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "TypedefUnion",
// CHECK-SAME:           baseType: ![[TD_UNION:.*]])
// CHECK: ![[TD_UNION]] = !DICompositeType(tag: DW_TAG_union_type,
// CHECK-SAME:             flags: DIFlagFwdDecl)

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "ObjCClass",
// CHECK-SAME:             scope: ![[MOD]],
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             runtimeLang: DW_LANG_ObjC)

// CHECK-NOT: !DICompositeType(tag: DW_TAG_structure_type,
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-SAME:             scope: ![[MOD]],
// CHECK-SAME:             flags: DIFlagFwdDecl)
