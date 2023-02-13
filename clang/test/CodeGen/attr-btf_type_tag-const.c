// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -S -emit-llvm -o - %s | FileCheck %s

// Check that BTF type tags are not attached to DW_TAG_const_type DIEs
// in presence of "sugar" expressions that are transparent for
// CGDebugInfo.cpp:UnwrapTypeForDebugInfo(), but are not transparent
// for local qualifiers.
//
// For details see:
//   CGDebugInfo::CreateType(const BTFTagAttributedType, llvm::DIFile)

#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))
#define __tag3 __attribute__((btf_type_tag("tag3")))

const int *foo;
typeof(*foo) __tag1 bar;

// CHECK: distinct !DIGlobalVariable(name: "bar", {{.*}}, type: ![[L01:[0-9]+]], {{.*}})
// CHECK: ![[L01]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L02:[0-9]+]])
// CHECK: ![[L02]] = !DIBasicType(name: "int", {{.*}}, annotations: ![[L03:[0-9]+]])
// CHECK: ![[L03]] = !{![[L04:[0-9]+]]}
// CHECK: ![[L04]] = !{!"btf:type_tag", !"tag1"}

const int __tag2 *buz;

// CHECK: distinct !DIGlobalVariable(name: "buz", {{.*}}, type: ![[L05:[0-9]+]], {{.*}})
// CHECK: ![[L05]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L06:[0-9]+]], {{.*}})
// CHECK: ![[L06]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L08:[0-9]+]])
// CHECK: ![[L08]] = !DIBasicType(name: "int", size: [[#]], {{.*}}, annotations: ![[L09:[0-9]+]])
// CHECK: ![[L09]] = !{![[L10:[0-9]+]]}
// CHECK: ![[L10]] = !{!"btf:type_tag", !"tag2"}

typeof(*buz) __tag3 quux;

// CHECK: distinct !DIGlobalVariable(name: "quux", {{.*}}, type: ![[L12:[0-9]+]], {{.*}})
// CHECK: ![[L12]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L13:[0-9]+]])
// CHECK: ![[L13]] = !DIBasicType(name: "int", {{.*}}, annotations: ![[L14:[0-9]+]])
// CHECK: ![[L14]] = !{![[L15:[0-9]+]], ![[L10]]}
// CHECK: ![[L15]] = !{!"btf:type_tag", !"tag3"}
