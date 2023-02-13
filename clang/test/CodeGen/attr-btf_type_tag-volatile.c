// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -S -emit-llvm -o - %s | FileCheck %s

// See attr-btf_type_tag-const.c for reasoning behind this test.
// Alternatively, see the following method:
//   CGDebugInfo::CreateType(const BTFTagAttributedType, llvm::DIFile)

#define __tag1 __attribute__((btf_type_tag("tag1")))

volatile int foo;
typeof(foo) __tag1 bar;

// CHECK: ![[#]] = distinct !DIGlobalVariable(name: "bar", {{.*}}, type: ![[L1:[0-9]+]], {{.*}})
// CHECK: ![[L1]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L2:[0-9]+]])
// CHECK: ![[L2]] = !DIBasicType(name: "int", size: [[#]], {{.*}}, annotations: ![[L3:[0-9]+]])
// CHECK: ![[L3]] = !{![[L4:[0-9]+]]}
// CHECK: ![[L4]] = !{!"btf:type_tag", !"tag1"}
