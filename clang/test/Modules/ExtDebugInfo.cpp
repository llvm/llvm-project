// RUN: rm -rf %t
// Test that only forward declarations are emitted for types dfined in modules.

// Modules:
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -debug-info-kind=limited \
// RUN:     -dwarf-ext-refs -fmodules                                   \
// RUN:     -fmodule-format=obj -fimplicit-module-maps -DMODULES \
// RUN:     -triple %itanium_abi_triple \
// RUN:     -fmodules-cache-path=%t %s -I %S/Inputs -I %t -emit-llvm -o %t-mod.ll
// RUN: cat %t-mod.ll |  FileCheck %s

// PCH:
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodule-format=obj -emit-pch -I%S/Inputs \
// RUN:     -triple %itanium_abi_triple \
// RUN:     -o %t.pch %S/Inputs/DebugCXX.h
// RUN: %clang_cc1 -std=c++11 -debug-info-kind=limited \
// RUN:     -dwarf-ext-refs -fmodule-format=obj \
// RUN:     -triple %itanium_abi_triple \
// RUN:     -include-pch %t.pch %s -emit-llvm -o %t-pch.ll %s
// RUN: cat %t-pch.ll |  FileCheck %s


#ifdef MODULES
@import DebugCXX;
#endif

using DebugCXX::Struct;

Struct s;
DebugCXX::Enum e;
DebugCXX::Template<long> implicitTemplate;
DebugCXX::Template<int> explicitTemplate;
DebugCXX::FloatInstatiation typedefTemplate;
int Struct::static_member = -1;
enum {
  e3 = -1
} conflicting_uid = e3;
auto anon_enum = DebugCXX::e2;
char _anchor = anon_enum + conflicting_uid;

TypedefUnion tdu;
TypedefEnum tde;
TypedefStruct tds;

InAnonymousNamespace anon;

void foo() {
  anon.i = GlobalStruct.i = GlobalUnion.i = GlobalEnum;
}

// CHECK: ![[NS:.*]] = !DINamespace(name: "DebugCXX", scope: ![[MOD:[0-9]+]],
// CHECK: ![[MOD]] = !DIModule(scope: null, name: {{.*}}DebugCXX

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Struct",
// CHECK-SAME:             scope: ![[NS]],
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX6StructE")

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum",
// CHECK-SAME:             scope: ![[NS]],
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier:  "_ZTSN8DebugCXX4EnumE")

// CHECK: !DICompositeType(tag: DW_TAG_class_type,

// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<int, DebugCXX::traits<int> >",
// CHECK-SAME:             scope: ![[NS]],
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIiNS_6traitsIiEEEE")

// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<float, DebugCXX::traits<float> >",
// CHECK-SAME:             scope: ![[NS]],
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIfNS_6traitsIfEEEE")

// CHECK: !DICompositeType(tag: DW_TAG_union_type,
// CHECK-SAME:             flags: DIFlagFwdDecl, identifier: "_ZTS12TypedefUnion")
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-SAME:             flags: DIFlagFwdDecl, identifier: "_ZTS11TypedefEnum")
// CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:             flags: DIFlagFwdDecl, identifier: "_ZTS13TypedefStruct")

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "static_member",
// CHECK-SAME:           scope: !"_ZTSN8DebugCXX6StructE"

// CHECK: !DIGlobalVariable(name: "anon_enum", {{.*}}, type: ![[ANON_ENUM:[0-9]+]]
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, scope: ![[NS]],
// CHECK-SAME:             line: 16

// CHECK: !DIGlobalVariable(name: "GlobalUnion",
// CHECK-SAME:              type: ![[GLOBAL_UNION:[0-9]+]]
// CHECK: ![[GLOBAL_UNION]] = !DICompositeType(tag: DW_TAG_union_type,
// CHECK-SAME:                elements: !{{[0-9]+}})
// CHECK: !DIGlobalVariable(name: "GlobalStruct",
// CHECK-SAME:              type: ![[GLOBAL_STRUCT:[0-9]+]]
// CHECK: ![[GLOBAL_STRUCT]] = !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:                elements: !{{[0-9]+}})

// CHECK: !DIGlobalVariable(name: "anon",
// CHECK-SAME:              type: ![[GLOBAL_ANON:[0-9]+]]
// CHECK: ![[GLOBAL_ANON]] = !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:              name: "InAnonymousNamespace", {{.*}}DIFlagFwdDecl)


// CHECK: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !"_ZTSN8DebugCXX6StructE", line: 27)

// CHECK: !DICompileUnit(
// CHECK-SAME:           splitDebugFilename:
// CHECK-SAME:           dwoId:
