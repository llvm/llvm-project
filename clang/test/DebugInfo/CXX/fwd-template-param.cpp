// RUN: %clang_cc1 %s -triple=%itanium_abi_triple -debug-info-kind=limited -debug-forward-template-params -emit-llvm -o - | FileCheck --check-prefix=CHILD %s
// RUN: %clang_cc1 %s -triple=%itanium_abi_triple -debug-info-kind=limited -emit-llvm -o - | FileCheck --check-prefix=NONE %s
// A DWARF forward declaration of a template instantiation should have template
// parameter children (if we ask for them).

template<typename T> class A;
A<int> *p;

// CHILD:      !DICompositeType(tag: DW_TAG_class_type, name: "A<int>"
// CHILD-SAME:     flags: DIFlagFwdDecl
// CHILD-SAME:     templateParams: [[PARAM_LIST:![0-9]*]]
// CHILD:      [[PARAM_LIST]] = !{[[PARAM:![0-9]*]]}
// CHILD:      [[PARAM]] = !DITemplateTypeParameter(name: "T",
// CHILD-SAME:     type: [[BTYPE:![0-9]*]]
// CHILD:      [[BTYPE]] = !DIBasicType(name: "int"

// NONE:       !DICompositeType(tag: DW_TAG_class_type, name: "A<int>"
// NONE-SAME:      flags: DIFlagFwdDecl
// NONE-NOT:       templateParams:
// NONE-SAME:      )
