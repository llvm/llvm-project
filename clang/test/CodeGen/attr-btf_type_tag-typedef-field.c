// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

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

// CHECK: ![[L4:[0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L16:[0-9]+]])
// CHECK: ![[L16]] = !{![[L17:[0-9]+]], ![[L24:[0-9]+]], ![[L31:[0-9]+]]}
// CHECK: ![[L17]] = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L18:[0-9]+]], size: [[#]])
// CHECK: ![[L18]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L19:[0-9]+]], size: [[#]], annotations: ![[L22:[0-9]+]])
// CHECK: ![[L19]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L4]], size: [[#]], annotations: ![[L20:[0-9]+]])
// CHECK: ![[L20]] = !{![[L21:[0-9]+]]}
// CHECK: ![[L21]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L22]] = !{![[L23:[0-9]+]]}
// CHECK: ![[L23]] = !{!"btf_type_tag", !"tag2"}
// CHECK: ![[L24]] = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L25:[0-9]+]]
// CHECK: ![[L25]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__fn2_t", file: ![[#]], line: [[#]], baseType: ![[L26:[0-9]+]])
// CHECK: ![[L26]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L27:[0-9]+]], size: [[#]], annotations: ![[L30:[0-9]+]])
// CHECK: ![[L27]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__fn_t", file: ![[#]], line: [[#]], baseType: ![[L28:[0-9]+]])
// CHECK: ![[L28]] = !DISubroutineType(types: ![[L29:[0-9]+]])
// CHECK: ![[L29]] = !{null, ![[L4]]}
// CHECK: ![[L30]] = !{![[L21]], ![[L23]]}
// CHECK: ![[L31]] = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: ![[#]], file: ![[#]], line: [[#]]1, baseType: ![[L32:[0-9]+]]
// CHECK: ![[L32]] = !DIBasicType(name: "long", size: [[#]], encoding: DW_ATE_signed)
