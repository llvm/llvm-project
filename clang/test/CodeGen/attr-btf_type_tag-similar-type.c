// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK-V2 %s

struct map_value {
        int __attribute__((btf_type_tag("tag1"))) __attribute__((btf_type_tag("tag3"))) *a;
        int __attribute__((btf_type_tag("tag2"))) __attribute__((btf_type_tag("tag4"))) *b;
};

struct map_value *func(void);

int test(struct map_value *arg)
{
        return *arg->a;
}

// CHECK: ![[L05:[0-9]+]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed)
// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_value", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L01:[0-9]+]])
// CHECK: ![[L01]] = !{![[L02:[0-9]+]], ![[L03:[0-9]+]]}
// CHECK: ![[L02]] = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L04:[0-9]+]], size: [[#]])
// CHECK: ![[L04]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L05]], size: [[#]], annotations: ![[L06:[0-9]+]])
// CHECK: ![[L06]] = !{![[L10:[0-9]+]], ![[L11:[0-9]+]]}
// CHECK: ![[L10]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L11]] = !{!"btf_type_tag", !"tag3"}
// CHECK: ![[L03]] = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L12:[0-9]+]], size: [[#]], offset: [[#]])
// CHECK: ![[L12]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L05]], size: [[#]], annotations: ![[L14:[0-9]+]])
// CHECK: ![[L14]] = !{![[L18:[0-9]+]], ![[L19:[0-9]+]]}
// CHECK: ![[L18]] = !{!"btf_type_tag", !"tag2"}
// CHECK: ![[L19]] = !{!"btf_type_tag", !"tag4"}

// CHECK-V2: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_value", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L01:[0-9]+]])
// CHECK-V2: ![[L01]] = !{![[L02:[0-9]+]], ![[L03:[0-9]+]]}
// CHECK-V2: ![[L02]] = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L04:[0-9]+]], size: [[#]])
// CHECK-V2: ![[L04]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L05:[0-9]+]], size: [[#]])
// CHECK-V2: ![[L05]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed, annotations: ![[L07:[0-9]+]])
// CHECK-V2: ![[L07]] = !{![[L08:[0-9]+]], ![[L09:[0-9]+]]}
// CHECK-V2: ![[L08]] = !{!"btf:type_tag", !"tag1"}
// CHECK-V2: ![[L09]] = !{!"btf:type_tag", !"tag3"}
// CHECK-V2: ![[L03]] = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L12:[0-9]+]], size: [[#]], offset: [[#]])
// CHECK-V2: ![[L12]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L13:[0-9]+]], size: [[#]])
// CHECK-V2: ![[L13]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed, annotations: ![[L15:[0-9]+]])
// CHECK-V2: ![[L15]] = !{![[L16:[0-9]+]], ![[L17:[0-9]+]]}
// CHECK-V2: ![[L16]] = !{!"btf:type_tag", !"tag2"}
// CHECK-V2: ![[L17]] = !{!"btf:type_tag", !"tag4"}
