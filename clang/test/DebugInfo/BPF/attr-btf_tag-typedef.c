// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_decl_tag("tag1")))
typedef struct { int a; } __s __tag1;
typedef unsigned * __u __tag1;
__s a;
__u u;

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__u", file: ![[#]], line: [[#]], baseType: ![[#]], annotations: ![[ANNOT:[0-9]+]])
// CHECK: ![[ANNOT]] = !{![[TAG1:[0-9]+]]}
// CHECK: ![[TAG1]] = !{!"btf_decl_tag", !"tag1"}

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__s", file: ![[#]], line: [[#]], baseType: ![[#]], annotations: ![[ANNOT]])
