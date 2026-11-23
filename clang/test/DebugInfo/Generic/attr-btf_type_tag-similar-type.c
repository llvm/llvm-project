// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -DDOUBLE_BRACKET_ATTRS=1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

#if DOUBLE_BRACKET_ATTRS
#define __tag1 [[clang::btf_type_tag("tag1")]]
#define __tag2 [[clang::btf_type_tag("tag2")]]
#define __tag3 [[clang::btf_type_tag("tag3")]]
#define __tag4 [[clang::btf_type_tag("tag4")]]
#else
#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))
#define __tag3 __attribute__((btf_type_tag("tag3")))
#define __tag4 __attribute__((btf_type_tag("tag4")))
#endif

struct map_value {
        int __tag1 __tag3 *a;
        int __tag2 __tag4 *b;
};

struct map_value *func(void);

int test(struct map_value *arg)
{
        return *arg->a;
}

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_value", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L14:[0-9]+]]
// CHECK: ![[L14]] = !{![[L15:[0-9]+]], ![[L20:[0-9]+]]}
// CHECK: ![[L15]] = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L16:[0-9]+]]
// CHECK: ![[L16]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[#]], size: [[#]], annotations: ![[L17:[0-9]+]]
// CHECK: ![[L17]] = !{![[L18:[0-9]+]], ![[L19:[0-9]+]]}
// CHECK: ![[L18]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L19]] = !{!"btf_type_tag", !"tag3"}
// CHECK: ![[L20]] = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L21:[0-9]+]]
// CHECK: ![[L21:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[#]], size: [[#]], annotations: ![[L22:[0-9]+]]
// CHECK: ![[L22]] = !{![[L23:[0-9]+]], ![[L24:[0-9]+]]}
// CHECK: ![[L23]] = !{!"btf_type_tag", !"tag2"}
// CHECK: ![[L24]] = !{!"btf_type_tag", !"tag4"}
