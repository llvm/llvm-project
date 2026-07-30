// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -gtemplate-alias %s -gsimple-template-names=simple \
// RUN: | FileCheck %s

//// Check that -gtemplate-alias causes DW_TAG_template_alias emission for
//// template aliases with default parameter values. See template-alias.cpp for
////  more template alias tests.
//// FIXME: We currently do not emit defaulted arguments.

template<typename T>
struct X {
  char m;
};

template<typename T>
struct Y {
  char n;
};

template <typename NonDefault, template <typename C> class T = Y, int I = 5, typename... Ts>
using A = X<NonDefault>;

//// We should be able to emit type alias metadata which describes all the
//// values, including the defaulted parameters and empty parameter pack.
A<int> a;

// CHECK: !DIDerivedType(tag: DW_TAG_template_alias, name: "A", file: ![[#]], line: [[#]], baseType: ![[baseType:[0-9]+]], extraData: ![[extraData:[0-9]+]])
// CHECK: ![[baseType]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X",
// CHECK: ![[int:[0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK: ![[extraData]] = !{![[NonDefault:[0-9]+]]}
// CHECK: ![[NonDefault]] = !DITemplateTypeParameter(name: "NonDefault", type: ![[int]])

//// FIXME: Ideally, we would describe the deafulted args, like this:
// : ![[extraData]] = !{![[NonDefault:[0-9]+]], ![[T:[0-9]+]], ![[I:[0-9]+]], ![[Ts:[0-9]+]]}
// : ![[NonDefault]] = !DITemplateTypeParameter(name: "NonDefault", type: ![[int]])
// : ![[T]] = !DITemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "T", defaulted: true, value: !"Y")
// : ![[I]] = !DITemplateValueParameter(name: "I", type: ![[int]], defaulted: true, value: i32 5)
// : ![[Ts]] = !DITemplateValueParameter(tag: DW_TAG_GNU_template_parameter_pack, name: "Ts", value: ![[types:[0-9]+]])
// : ![[types]] = !{}
