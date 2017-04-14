// RUN: rm -rf %t
// Test that only forward declarations are emitted for types defined in modules.

// Modules:
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -debug-info-kind=standalone \
// RUN:     -dwarf-ext-refs -fmodules                                   \
// RUN:     -fmodule-format=obj -fimplicit-module-maps -DMODULES \
// RUN:     -triple %itanium_abi_triple \
// RUN:     -fmodules-cache-path=%t %s -I %S/Inputs -I %t -emit-llvm -o %t-mod.ll
// RUN: cat %t-mod.ll |  FileCheck %s

// PCH:
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodule-format=obj -emit-pch -I%S/Inputs \
// RUN:     -triple %itanium_abi_triple \
// RUN:     -o %t.pch %S/Inputs/DebugCXX.h
// RUN: %clang_cc1 -std=c++11 -debug-info-kind=standalone \
// RUN:     -dwarf-ext-refs -fmodule-format=obj \
// RUN:     -triple %itanium_abi_triple \
// RUN:     -include-pch %t.pch %s -emit-llvm -o %t-pch.ll
// RUN: cat %t-pch.ll |  FileCheck %s
// RUN: cat %t-pch.ll |  FileCheck %s --check-prefix=CHECK-PCH

#ifdef MODULES
@import DebugCXX;
#endif

using DebugCXX::Struct;

Struct s;
DebugCXX::Enum e;

// Template instantiations.
DebugCXX::Template<long> implicitTemplate;
DebugCXX::Template<int> explicitTemplate;
DebugCXX::FloatInstantiation typedefTemplate;
DebugCXX::B anchoredTemplate;

int Struct::static_member = -1;
enum {
  e3 = -1
} conflicting_uid = e3;
auto anon_enum = DebugCXX::e2;
char _anchor = anon_enum + conflicting_uid;

TypedefUnion tdu;
TypedefEnum tde;
TypedefStruct tds;
TypedefTemplate tdt;
Template1<int> explicitTemplate1;

template <class T> class FwdDeclTemplate { T t; };
TypedefFwdDeclTemplate tdfdt;

InAnonymousNamespace anon;

// Types that are forward-declared in the module and defined here.
struct PureFwdDecl { int i; };
PureFwdDecl definedLocally;

struct Specialized<int>::Member { int i; };
struct Specialized<int>::Member definedLocally2;

template <class T> struct FwdDeclTemplateMember<T>::Member { T t; };
TypedefFwdDeclTemplateMember tdfdtm;

SpecializedBase definedLocally3;
extern template class WithSpecializedBase<int>;
WithSpecializedBase<int> definedLocally4;

void foo() {
  anon.i = GlobalStruct.i = GlobalUnion.i = GlobalEnum;
}

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum",
// CHECK-SAME:             scope: ![[NS:[0-9]+]],
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier:  "_ZTSN8DebugCXX4EnumE")

// CHECK: ![[NS]] = !DINamespace(name: "DebugCXX", scope: ![[MOD:[0-9]+]],
// CHECK: ![[MOD]] = !DIModule(scope: null, name: {{.*}}DebugCXX

// This type is anchored in the module by an explicit template instantiation.
// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<long, DebugCXX::traits<long> >",
// CHECK-SAME:             scope: ![[NS]],
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIlNS_6traitsIlEEEE")

// This type is anchored in the module by an explicit template instantiation.
// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<int, DebugCXX::traits<int> >",
// CHECK-SAME:             scope: ![[NS]],
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIiNS_6traitsIiEEEE")

// This type isn't, however, even with standalone non-module debug info this
// type is a forward declaration.
// CHECK-NOT: !DICompositeType(tag: DW_TAG_structure_type, name: "traits<int>",

// This one isn't.
// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<float, DebugCXX::traits<float> >",
// CHECK-SAME:             scope: ![[NS]],
// CHECK-SAME:             elements:
// CHECK-SAME:             templateParams:
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIfNS_6traitsIfEEEE")

// This type is anchored in the module by an explicit template instantiation.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "traits<float>",
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX6traitsIfEE")


// This type is anchored in the module via a function argument,
// but we don't know this (yet).
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "A<void>",
// CHECK-SAME:             scope: ![[NS]],
// CHECK-SAME:             elements:
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX1AIJvEEE")

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "static_member",
// CHECK-SAME:           scope: ![[STRUCT:[0-9]*]]

// CHECK: ![[STRUCT]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Struct",
// CHECK-SAME:             scope: ![[NS]],
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX6StructE")

// CHECK: !DICompositeType(tag: DW_TAG_union_type,
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTS12TypedefUnion")
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTS11TypedefEnum")
// CHECK: !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTS13TypedefStruct")

// This one isn't.
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "Template1<void *>",
// CHECK-SAME:             elements:
// CHECK-SAME:             templateParams:
// CHECK-SAME:             identifier: "_ZTS9Template1IPvE")

// This type is anchored in the module by an explicit template instantiation.
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "Template1<int>",
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-SAME:             identifier: "_ZTS9Template1IiE")

// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "FwdDeclTemplate<int>",
// CHECK-SAME:             elements:
// CHECK-SAME:             templateParams:
// CHECK-SAME:             identifier: "_ZTS15FwdDeclTemplateIiE")

// This type is defined locally and forward-declared in the module.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "PureFwdDecl",
// CHECK-SAME:             elements:
// CHECK-SAME:             identifier: "_ZTS11PureFwdDecl")

// This type is defined locally and forward-declared in the module.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Member",
// CHECK-SAME:             elements:
// CHECK-SAME:             identifier: "_ZTSN11SpecializedIiE6MemberE")

// This type is defined locally and forward-declared in the module.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Member",
// CHECK-SAME:             elements:
// CHECK-SAME:             identifier: "_ZTSN21FwdDeclTemplateMemberIiE6MemberE")

// This type is defined locally and forward-declared in the module.
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "SpecializedBase",
// CHECK-SAME:           baseType: ![[SPECIALIZEDBASE:.*]])
// CHECK: ![[SPECIALIZEDBASE]] =
// CHECK-SAME: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:                  name: "WithSpecializedBase<float>",
// CHECK-SAME:                  elements:
// CHECK-SAME:                  identifier: "_ZTS19WithSpecializedBaseIfE")

// This type is explicitly specialized locally.
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "WithSpecializedBase<int>",
// CHECK-SAME:             elements:
// CHECK-SAME:             identifier: "_ZTS19WithSpecializedBaseIiE")

// CHECK: !DIGlobalVariable(name: "anon_enum", {{.*}}, type: ![[ANON_ENUM:[0-9]+]]
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, scope: ![[NS]],
// CHECK-SAME:             line: 16

// CHECK: !DIGlobalVariable(name: "GlobalUnion",
// CHECK-SAME:              type: ![[GLOBAL_UNION:[0-9]+]]
// CHECK: ![[GLOBAL_UNION]] = distinct !DICompositeType(tag: DW_TAG_union_type,
// CHECK-SAME:                elements: !{{[0-9]+}})
// CHECK: !DIGlobalVariable(name: "GlobalStruct",
// CHECK-SAME:              type: ![[GLOBAL_STRUCT:[0-9]+]]
// CHECK: ![[GLOBAL_STRUCT]] = distinct !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:                elements: !{{[0-9]+}})


// CHECK: !DIGlobalVariable(name: "anon",
// CHECK-SAME:              type: ![[GLOBAL_ANON:[0-9]+]]
// CHECK: ![[GLOBAL_ANON]] = !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:              name: "InAnonymousNamespace", {{.*}}DIFlagFwdDecl)


// CHECK: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !{{[0-9]+}}, entity: ![[STRUCT]], line: 27)

// CHECK: !DICompileUnit(
// CHECK-SAME:           splitDebugFilename:
// CHECK-SAME:           dwoId:
// CHECK-PCH: !DICompileUnit({{.*}}splitDebugFilename:
// CHECK-PCH:                dwoId: 18446744073709551614
