// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_type_tag("tag1")))
void __tag1 *g;

// CHECK: distinct !DIGlobalVariable(name: "g", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L1:[0-9]+]], isLocal: false, isDefinition: true)
// CHECK: ![[L1]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L2:[0-9]+]], size: [[#]])
// CHECK: ![[L2]] = !DIBasicType(tag: DW_TAG_unspecified_type, name: "void", annotations: ![[L4:[0-9]+]])
// CHECK: ![[L4]] = !{![[L5:[0-9]+]]}
// CHECK: ![[L5]] = !{!"btf:type_tag", !"tag1"}
