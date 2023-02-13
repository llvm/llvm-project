// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -emit-llvm -o - %s | FileCheck %s --check-prefixes CHECK-V2

#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))

typedef void __fn_t(int);
typedef __fn_t __tag1 __tag2 *__fn2_t;
struct t {
  int __tag1 * __tag2 *a;
  __fn2_t b;
  long c;
};
int *foo1(struct t *a1) {
  return (int *)a1->c;
}

// CHECK: ![[L01:[0-9]+]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed)
// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L02:[0-9]+]])
// CHECK: ![[L02]] = !{![[L03:[0-9]+]], ![[L04:[0-9]+]], ![[L05:[0-9]+]]}
// CHECK: ![[L03]] = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L06:[0-9]+]], size: [[#]])
// CHECK: ![[L06]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L07:[0-9]+]], size: [[#]], annotations: ![[L08:[0-9]+]])
// CHECK: ![[L07]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L01]], size: [[#]], annotations: ![[L10:[0-9]+]])
// CHECK: ![[L10]] = !{![[L14:[0-9]+]]}
// CHECK: ![[L14]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L08]] = !{![[L15:[0-9]+]]}
// CHECK: ![[L15]] = !{!"btf_type_tag", !"tag2"}
// CHECK: ![[L04]] = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L16:[0-9]+]], size: [[#]], offset: [[#]])
// CHECK: ![[L16]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__fn2_t", file: ![[#]], line: [[#]], baseType: ![[L17:[0-9]+]])
// CHECK: ![[L17]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L18:[0-9]+]], size: [[#]], annotations: ![[L19:[0-9]+]])
// CHECK: ![[L18]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__fn_t", file: ![[#]], line: [[#]], baseType: ![[L20:[0-9]+]])
// CHECK: ![[L20]] = !DISubroutineType(types: ![[L22:[0-9]+]])
// CHECK: ![[L22]] = !{null, ![[L01]]}
// CHECK: ![[L19]] = !{![[L14]], ![[L15]]}
// CHECK: ![[L05]] = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L23:[0-9]+]], size: [[#]], offset: [[#]])
// CHECK: ![[L23]] = !DIBasicType(name: "long", size: [[#]], encoding: DW_ATE_signed)

// CHECK-V2: ![[L01:[0-9]+]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed)
// CHECK-V2: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L02:[0-9]+]])
// CHECK-V2: ![[L02]] = !{![[L03:[0-9]+]], ![[L04:[0-9]+]], ![[L05:[0-9]+]]}
// CHECK-V2: ![[L03]] = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L06:[0-9]+]], size: [[#]])
// CHECK-V2: ![[L06]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L07:[0-9]+]], size: [[#]])
// CHECK-V2: ![[L07]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L09:[0-9]+]], size: [[#]], annotations: ![[L08:[0-9]+]])
// CHECK-V2: ![[L09]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed, annotations: ![[L10:[0-9]+]])
// CHECK-V2: ![[L10]] = !{![[L14:[0-9]+]]}
// CHECK-V2: ![[L14]] = !{!"btf:type_tag", !"tag1"}
// CHECK-V2: ![[L08]] = !{![[L15:[0-9]+]]}
// CHECK-V2: ![[L15]] = !{!"btf:type_tag", !"tag2"}
// CHECK-V2: ![[L04]] = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L16:[0-9]+]], size: [[#]], offset: [[#]])
// CHECK-V2: ![[L16]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__fn2_t", file: ![[#]], line: [[#]], baseType: ![[L17:[0-9]+]])
// CHECK-V2: ![[L17]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L18:[0-9]+]], size: [[#]])
// CHECK-V2: ![[L18]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__fn_t", file: ![[#]], line: [[#]], baseType: ![[L20:[0-9]+]], annotations: ![[L19:[0-9]+]])
// CHECK-V2: ![[L20]] = !DISubroutineType(types: ![[L22:[0-9]+]])
// CHECK-V2: ![[L22]] = !{null, ![[L01]]}
// CHECK-V2: ![[L19]] = !{![[L14]], ![[L15]]}
// CHECK-V2: ![[L05]] = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L23:[0-9]+]], size: [[#]], offset: [[#]])
// CHECK-V2: ![[L23]] = !DIBasicType(name: "long", size: [[#]], encoding: DW_ATE_signed)
