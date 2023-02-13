// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_type_tag("tag1")))

struct st {
  struct st __tag1 *self;
} g;

// CHECK: distinct !DIGlobalVariable(name: "g", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L1:[0-9]+]], isLocal: false, isDefinition: true)
// CHECK: ![[L1]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "st", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L2:[0-9]+]])
// CHECK: ![[L2]] = !{![[L3:[0-9]+]]}
// CHECK: ![[L3]] = !DIDerivedType(tag: DW_TAG_member, name: "self", scope: ![[L1]], file: ![[#]], line: [[#]], baseType: ![[L4:[0-9]+]], size: [[#]])
// CHECK: ![[L4]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L5:[0-9]+]], size: [[#]])
// CHECK: ![[L5]] = !DICompositeType(tag: DW_TAG_structure_type, name: "st", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L2]], annotations: ![[L7:[0-9]+]])
// CHECK: ![[L7]] = !{![[L8:[0-9]+]]}
// CHECK: ![[L8]] = !{!"btf:type_tag", !"tag1"}
