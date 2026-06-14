// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -disable-llvm-passes -debug-info-kind=limited %s -o - | FileCheck %s

#define __tag1 __attribute__((btf_type_tag("tag1")))
int _Atomic __tag1 *g1;
volatile int _Atomic __tag1 *g2;

// CHECK: distinct !DIGlobalVariable(name: "g1", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[PTR1:[0-9]+]]
// CHECK: distinct !DIGlobalVariable(name: "g2", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[PTR2:[0-9]+]]
// CHECK: ![[PTR2]]  = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[BASE2:[0-9]+]], size: [[#]], annotations: ![[ANNOT:[0-9]+]])
// CHECK: ![[BASE2]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[BASE1:[0-9]+]])
// CHECK: ![[BASE1]] = !DIDerivedType(tag: DW_TAG_atomic_type, baseType: ![[BASIC:[0-9]+]])
// CHECK: ![[BASIC]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed)
// CHECK: ![[ANNOT]] = !{![[ENTRY:[0-9]+]]}
// CHECK: ![[ENTRY]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[PTR1]]  = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[BASE1]], size: [[#]], annotations: ![[ANNOT]])
