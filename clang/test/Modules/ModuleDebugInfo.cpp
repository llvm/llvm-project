// Test that (the same) debug info is emitted for an Objective-C++
// module and a C++ precompiled header.

// REQUIRES: asserts

// Modules:
// RUN: rm -rf %t
// RUN: %clang_cc1 -triple %itanium_abi_triple -x objective-c++ -std=c++11 -debugger-tuning=lldb -debug-info-kind=limited -fmodules -fmodule-format=obj -fimplicit-module-maps -DMODULES -fmodules-cache-path=%t %s -I %S/Inputs -I %t -emit-llvm -o %t.ll -mllvm -debug-only=pchcontainer &>%t-mod.ll
// RUN: cat %t-mod.ll | FileCheck %s
// RUN: cat %t-mod.ll | FileCheck --check-prefix=CHECK-NEG %s
// RUN: cat %t-mod.ll | FileCheck --check-prefix=CHECK-MOD %s

// PCH:
// RUN: %clang_cc1 -triple %itanium_abi_triple -x c++ -std=c++11  -debugger-tuning=lldb -emit-pch -fmodule-format=obj -I %S/Inputs -o %t.pch %S/Inputs/DebugCXX.h -mllvm -debug-only=pchcontainer &>%t-pch.ll
// RUN: cat %t-pch.ll | FileCheck %s
// RUN: cat %t-pch.ll | FileCheck --check-prefix=CHECK-NEG %s

#ifdef MODULES
@import DebugCXX;
#endif

// CHECK-MOD: distinct !DICompileUnit(language: DW_LANG_{{.*}}C_plus_plus,
// CHECK-MOD: distinct !DICompileUnit(language: DW_LANG_{{.*}}C_plus_plus,

// CHECK: distinct !DICompileUnit(language: DW_LANG_{{.*}}C_plus_plus,
// CHECK-SAME:                    isOptimized: false,
// CHECK-NOT:                     splitDebugFilename:
// CHECK-SAME:                    dwoId:

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum"
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX4EnumE")
// CHECK: !DINamespace(name: "DebugCXX"

// CHECK-MOD: ![[DEBUGCXX:.*]] = !DIModule(scope: null, name: "DebugCXX

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-NOT:              name:
// CHECK-SAME:             )

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-NOT:              name:
// CHECK-SAME:             )

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-NOT:              name:
// CHECK-SAME:             identifier: "_ZTS11TypedefEnum")

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-NOT:              name:
// CHECK-SAME:             )
// CHECK: !DIEnumerator(name: "e5", value: 5, isUnsigned: true)

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "B",
// no mangled name here yet.

// This type is anchored by a function parameter.
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "A<void>"
// CHECK-SAME:             elements:
// CHECK-SAME:             templateParams:
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX1AIJvEEE")

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Struct"
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX6StructE")

// This type is anchored by an explicit template instantiation.
// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<int, DebugCXX::traits<int> >"
// CHECK-SAME:             elements:
// CHECK-SAME:             templateParams:
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIiNS_6traitsIiEEEE")

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "traits<int>"
// CHECK-SAME:             flags: DIFlagFwdDecl
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX6traitsIiEE")

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "traits<float>"
// CHECK-SAME:             elements:
// CHECK-SAME:             templateParams:
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX6traitsIfEE")

// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<long, DebugCXX::traits<long> >"
// CHECK-SAME:             elements:
// CHECK-SAME:             templateParams:
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIlNS_6traitsIlEEEE")

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "FloatInstantiation"
// no mangled name here yet.

// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<float, DebugCXX::traits<float> >"
// CHECK-SAME:             flags: DIFlagFwdDecl
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIfNS_6traitsIfEEEE")

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Virtual"
// CHECK-SAME:             elements:
// CHECK-SAME:             identifier: "_ZTS7Virtual")
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "_vptr$Virtual"

// CHECK: !DICompositeType(tag: DW_TAG_union_type,
// CHECK-NOT:              name:
// CHECK-SAME:             identifier: "_ZTS12TypedefUnion")

// CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-NOT:              name:
// CHECK-SAME:             identifier: "_ZTS13TypedefStruct")

// CHECK: !DICompositeType(tag: DW_TAG_union_type,
// CHECK-NOT:              name:
// CHECK-SAME:             )

// CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-NOT:              name:
// CHECK-SAME:             )

// CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:             name: "InAnonymousNamespace",
// CHECK-SAME:             elements: !{{[0-9]+}})

// CHECK: ![[A:.*]] = {{.*}}!DICompositeType(tag: DW_TAG_class_type, name: "A",
// CHECK-SAME:                               elements:
// CHECK-SAME:                               vtableHolder: ![[A]])

// CHECK: ![[DERIVED:.*]] = {{.*}}!DICompositeType(tag: DW_TAG_class_type, name: "Derived",
// CHECK-SAME:                                     identifier: "_ZTS7Derived")
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "B", scope: ![[DERIVED]],
// CHECK-SAME:             elements: ![[B_MBRS:.*]], vtableHolder:
// CHECK: ![[B_MBRS]] = !{{{.*}}, ![[GET_PARENT:.*]]}
// CHECK: ![[GET_PARENT]] = !DISubprogram(name: "getParent"

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "TypedefTemplate",
// CHECK-SAME:           baseType: ![[BASE:.*]])
// CHECK: ![[BASE]] = !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:                         name: "Template1<void *>",
// CHECK-SAME:                         flags: DIFlagFwdDecl,
// CHECK-SAME:                         identifier: "_ZTS9Template1IPvE")

// Explicit instantiation.
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "Template1<int>",
// CHECK-SAME:             templateParams:
// CHECK-SAME:             identifier: "_ZTS9Template1IiE")

// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "FwdDeclTemplate<int>",
// CHECK-SAME:             flags: DIFlagFwdDecl
// CHECK-SAME:             identifier: "_ZTS15FwdDeclTemplateIiE")

// Forward-declared member of a template.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Member",
// CHECK-SAME:             flags: DIFlagFwdDecl
// CHECK-SAME:             identifier: "_ZTSN21FwdDeclTemplateMemberIiE6MemberE")

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "SpecializedBase",
// CHECK-SAME:           baseType: ![[SPECIALIZEDBASE:.*]])
// CHECK: ![[SPECIALIZEDBASE]] = !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:                             name: "WithSpecializedBase<float>",
// CHECK-SAME:                             flags: DIFlagFwdDecl,

// CHECK-MOD: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[DEBUGCXX]],
// CHECK-MOD-SAME:              entity: ![[DUMMY:[0-9]+]],
// CHECK-MOD-SAME:              line: 3)
// CHECK-MOD: ![[DUMMY]] = !DIModule(scope: null, name: "dummy",
// CHECK-MOD: distinct !DICompileUnit(language: DW_LANG_ObjC_plus_plus,
// CHECK-MOD-SAME:  splitDebugFilename: "{{.*}}dummy{{.*}}.pcm",

// CHECK-NEG-NOT: !DICompositeType(tag: DW_TAG_structure_type, name: "PureForwardDecl"
