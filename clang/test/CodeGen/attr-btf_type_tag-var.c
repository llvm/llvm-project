// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -DDOUBLE_BRACKET_ATTRS=1 \
// RUN:   -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefixes CHECK-V2 %s
// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -DDOUBLE_BRACKET_ATTRS=1 -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefixes CHECK-V2 %s

#if DOUBLE_BRACKET_ATTRS
#define __tag1 [[clang::btf_type_tag("tag1")]]
#define __tag2 [[clang::btf_type_tag("tag2")]]
#define __tag3 [[clang::btf_type_tag("tag3")]]
#define __tag4 [[clang::btf_type_tag("tag4")]]
#define __tag5 [[clang::btf_type_tag("tag5")]]
#define __tag6 [[clang::btf_type_tag("tag6")]]

const volatile int __tag1 __tag2 * __tag3 __tag4 const volatile  * __tag5 __tag6 const volatile * g;
#else
#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))
#define __tag3 __attribute__((btf_type_tag("tag3")))
#define __tag4 __attribute__((btf_type_tag("tag4")))
#define __tag5 __attribute__((btf_type_tag("tag5")))
#define __tag6 __attribute__((btf_type_tag("tag6")))

const int __tag1 __tag2 volatile * const __tag3  __tag4  volatile * __tag5  __tag6 const volatile * g;
#endif

// CHECK: distinct !DIGlobalVariable(name: "g", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L01:[0-9]+]], isLocal: false, isDefinition: true)
// CHECK: ![[L01]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L02:[0-9]+]], size: [[#]], annotations: ![[L03:[0-9]+]])
// CHECK: ![[L02]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L04:[0-9]+]])
// CHECK: ![[L04]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L05:[0-9]+]])
// CHECK: ![[L05]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L06:[0-9]+]], size: [[#]], annotations: ![[L07:[0-9]+]])
// CHECK: ![[L06]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L08:[0-9]+]])
// CHECK: ![[L08]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L09:[0-9]+]])
// CHECK: ![[L09]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L10:[0-9]+]], size: [[#]], annotations: ![[L11:[0-9]+]])
// CHECK: ![[L10]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L12:[0-9]+]])
// CHECK: ![[L12]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L13:[0-9]+]])
// CHECK: ![[L13]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed)
// CHECK: ![[L11]] = !{![[L19:[0-9]+]], ![[L20:[0-9]+]]}
// CHECK: ![[L19]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L20]] = !{!"btf_type_tag", !"tag2"}
// CHECK: ![[L07]] = !{![[L23:[0-9]+]], ![[L24:[0-9]+]]}
// CHECK: ![[L23]] = !{!"btf_type_tag", !"tag3"}
// CHECK: ![[L24]] = !{!"btf_type_tag", !"tag4"}
// CHECK: ![[L03]] = !{![[L25:[0-9]+]], ![[L26:[0-9]+]]}
// CHECK: ![[L25]] = !{!"btf_type_tag", !"tag5"}
// CHECK: ![[L26]] = !{!"btf_type_tag", !"tag6"}

// CHECK-V2: distinct !DIGlobalVariable(name: "g", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L01:[0-9]+]], isLocal: false, isDefinition: true)
// CHECK-V2: ![[L01]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L02:[0-9]+]], size: [[#]])
// CHECK-V2: ![[L02]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L04:[0-9]+]])
// CHECK-V2: ![[L04]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L05:[0-9]+]])
// CHECK-V2: ![[L05]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L06:[0-9]+]], size: [[#]], annotations: ![[L07:[0-9]+]])
// CHECK-V2: ![[L06]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L08:[0-9]+]])
// CHECK-V2: ![[L08]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L09:[0-9]+]])
// CHECK-V2: ![[L09]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L10:[0-9]+]], size: [[#]], annotations: ![[L11:[0-9]+]])
// CHECK-V2: ![[L10]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L12:[0-9]+]])
// CHECK-V2: ![[L12]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L13:[0-9]+]])
// CHECK-V2: ![[L13]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed, annotations: ![[L14:[0-9]+]])
// CHECK-V2: ![[L14]] = !{![[L15:[0-9]+]], ![[L16:[0-9]+]]}
// CHECK-V2: ![[L15]] = !{!"btf:type_tag", !"tag1"}
// CHECK-V2: ![[L16]] = !{!"btf:type_tag", !"tag2"}
// CHECK-V2: ![[L11]] = !{![[L17:[0-9]+]], ![[L18:[0-9]+]]}
// CHECK-V2: ![[L17]] = !{!"btf:type_tag", !"tag3"}
// CHECK-V2: ![[L18]] = !{!"btf:type_tag", !"tag4"}
// CHECK-V2: ![[L07]] = !{![[L21:[0-9]+]], ![[L22:[0-9]+]]}
// CHECK-V2: ![[L21]] = !{!"btf:type_tag", !"tag5"}
// CHECK-V2: ![[L22]] = !{!"btf:type_tag", !"tag6"}
