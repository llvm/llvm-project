// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -DDOUBLE_BRACKET_ATTRS=1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

#if DOUBLE_BRACKET_ATTRS
#define __tag1 [[clang::btf_type_tag("tag1")]]
#define __tag2 [[clang::btf_type_tag("tag2")]]
#else
#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))
#endif

struct bar {
  int c;
};
typedef struct bar __tag1 __tag2 bar_t1;
typedef const struct bar __tag1 __tag2 bar_t2;
typedef volatile struct bar __tag1 __tag2 bar_t3;
typedef volatile struct bar * __tag1 __tag2 bar_t4;

typedef const int __tag1 __tag2 int_v;

int use(bar_t1 *v1, bar_t2 *v2, bar_t3 *v3, bar_t4 v4, int_v v5)
{
  return v1->c + v2->c + v3->c + v4->c + v5;
}

// CHECK: ![[L4:[0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK: ![[L10:[0-9]+]] = !DIDerivedType(tag: DW_TAG_typedef, name: "bar_t1", file: ![[#]], line: [[#]], baseType: ![[L11:[0-9]+]], annotations: ![[L14:[0-9]+]])
// CHECK: ![[L11]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: ![[#]], line: [[#]], size: [[#]], elements: ![[#]])
// CHECK: ![[L14]] = !{![[L15:[0-9]+]], ![[L16:[0-9]+]]}
// CHECK: ![[L15]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L16]] = !{!"btf_type_tag", !"tag2"}
// CHECK: ![[L18:[0-9]+]] = !DIDerivedType(tag: DW_TAG_typedef, name: "bar_t2", file: ![[#]], line: [[#]], baseType: ![[L19:[0-9]+]], annotations: ![[L14]])
// CHECK: ![[L19]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L11]])
// CHECK: ![[L21:[0-9]+]] = !DIDerivedType(tag: DW_TAG_typedef, name: "bar_t3", file: ![[#]], line: [[#]], baseType: ![[L22:[0-9]+]], annotations: ![[L14]])
// CHECK: ![[L22]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L11]])
// CHECK: ![[L23:[0-9]+]] = !DIDerivedType(tag: DW_TAG_typedef, name: "bar_t4", file: ![[#]], line: [[#]], baseType: ![[L24:[0-9]+]], annotations: ![[L14]])
// CHECK: ![[L24]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L22]], size: [[#]])
// CHECK: ![[L25:[0-9]+]] = !DIDerivedType(tag: DW_TAG_typedef, name: "int_v", file: ![[#]], line: [[#]], baseType: ![[L26:[0-9]+]], annotations: ![[L14]])
// CHECK: ![[L26]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L4]])
