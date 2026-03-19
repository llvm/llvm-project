// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -gtemplate-alias %s -gsimple-template-names=simple \
// RUN: | FileCheck %s

//// Check that -gtemplate-alias causes DW_TAG_template_alias emission for
//// variadic template aliases. See template-alias.cpp for more template alias
//// tests.

template<typename Y, int Z>
struct X {
  Y m1 = Z;
};

template<int I, typename... Ts>
using A = X<Ts..., I>;

A<5, int> a;

// CHECK: !DIDerivedType(tag: DW_TAG_template_alias, name: "A", file: ![[#]], line: [[#]], baseType: ![[baseType:[0-9]+]], extraData: ![[extraData:[0-9]+]])
// CHECK: ![[baseType]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X",
// CHECK: ![[int:[0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK: ![[extraData]] = !{![[I:[0-9]+]], ![[Ts:[0-9]+]]}
// CHECK: ![[I]] = !DITemplateValueParameter(name: "I", type: ![[int]], value: i32 5)
// CHECK: ![[Ts]] = !DITemplateValueParameter(tag: DW_TAG_GNU_template_parameter_pack, name: "Ts", value: ![[types:[0-9]+]])
// CHECK: ![[types]] = !{![[int_template_param:[0-9]+]]}
// CHECK: ![[int_template_param]] = !DITemplateTypeParameter(type: ![[int]])
