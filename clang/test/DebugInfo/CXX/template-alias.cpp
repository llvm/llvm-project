// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -gtemplate-alias %s -gsimple-template-names=simple \
// RUN: | FileCheck %s --check-prefixes=ALIAS-SIMPLE,ALIAS-ALL

// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -gtemplate-alias %s -gsimple-template-names=mangled \
// RUN: | FileCheck %s --check-prefixes=ALIAS-MANGLED,ALIAS-ALL

// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -gtemplate-alias %s  \
// RUN: | FileCheck %s --check-prefixes=ALIAS-FULL,ALIAS-ALL

// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone  %s \
// RUN: | FileCheck %s --check-prefixes=TYPEDEF


//// Check that -gtemplate-alias causes DW_TAG_template_alias emission for
//// template aliases, and that respects gsimple-template-names.
////
//// Test type and value template parameters.

template<typename Y, int Z>
struct X {
  Y m1 = Z;
};

template<typename B, int C>
using A = X<B, C>;

A<int, 5> a;


// ALIAS-SIMPLE: !DIDerivedType(tag: DW_TAG_template_alias, name: "A", file: ![[#]], line: [[#]], baseType: ![[baseType:[0-9]+]], extraData: ![[extraData:[0-9]+]])
// ALIAS-SIMPLE: ![[baseType]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X",

// FIXME: Mangled name is wrong (not a regression).
// ALIAS-MANGLED: !DIDerivedType(tag: DW_TAG_template_alias, name: "A<int, 5>", file: ![[#]], line: [[#]], baseType: ![[baseType:[0-9]+]], extraData: ![[extraData:[0-9]+]])
// ALIAS-MANGLED: ![[baseType]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_STN|X|<int, 5>",

// ALIAS-FULL: !DIDerivedType(tag: DW_TAG_template_alias, name: "A<int, 5>", file: ![[#]], line: [[#]], baseType: ![[baseType:[0-9]+]], extraData: ![[extraData:[0-9]+]])
// ALIAS-FULL: ![[baseType]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X<int, 5>",

// ALIAS-ALL: ![[int:[0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// ALIAS-ALL: ![[extraData]] = !{![[B:[0-9]+]], ![[C:[0-9]+]]}
// ALIAS-ALL: ![[B]] = !DITemplateTypeParameter(name: "B", type: ![[int]])
// ALIAS-ALL: ![[C]] = !DITemplateValueParameter(name: "C", type: ![[int]], value: i32 5)

// TYPEDEF: !DIDerivedType(tag: DW_TAG_typedef, name: "A<int, 5>", file: ![[#]], line: [[#]], baseType: ![[baseType:[0-9]+]])
// TYPEDEF: ![[baseType]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X<int, 5>",
// TYPEDEF: ![[int:[0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
