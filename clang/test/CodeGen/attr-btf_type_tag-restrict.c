// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -S -emit-llvm -o - %s | FileCheck %s

// See attr-btf_type_tag-const.c for reasoning behind this test.
// Alternatively, see the following method:
//   CGDebugInfo::CreateType(const BTFTagAttributedType, llvm::DIFile)

#define __tag1 __attribute__((btf_type_tag("tag1")))

void foo(int * restrict bar, typeof(bar) __tag1 buz) {}

// CHECK: ![[#]] = !DISubroutineType(types: ![[L1:[0-9]+]])
// CHECK: ![[L1]] = !{null, ![[L2:[0-9]+]], ![[L3:[0-9]+]]}
// CHECK: ![[L2]] = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: ![[L4:[0-9]+]])
// CHECK: ![[L4]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L5:[0-9]+]], {{.*}})
// CHECK: ![[L5]] = !DIBasicType(name: "int", {{.*}}, encoding: DW_ATE_signed)
// CHECK: ![[L3]] = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: ![[L6:[0-9]+]])
// CHECK: ![[L6]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L5]], {{.*}}, annotations: ![[L7:[0-9]+]])
// CHECK: ![[L7]] = !{![[L8:[0-9]+]]}
// CHECK: ![[L8]] = !{!"btf:type_tag", !"tag1"}
