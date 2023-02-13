// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 \
// RUN:   -triple %itanium_abi_triple -debug-info-kind=limited \
// RUN:   -mllvm -btf-type-tag-v2 -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix CHECK-V2 %s

struct t {
 int (__attribute__((btf_type_tag("rcu"))) *f)();
 int a;
};
int foo(struct t *arg) {
  return arg->a;
}

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "f", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L1:[0-9]+]], size: [[#]])
// CHECK: ![[L1]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[#]], size: [[#]], annotations: ![[L2:[0-9]+]])
// CHECK: ![[L2]] = !{![[L3:[0-9]+]]}
// CHECK: ![[L3]] = !{!"btf_type_tag", !"rcu"}

// CHECK-V2: !DIDerivedType(tag: DW_TAG_member, name: "f", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L1:[0-9]+]], size: [[#]])
// CHECK-V2: ![[L1]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L2:[0-9]+]], size: [[#]])
// CHECK-V2: ![[L2]] = !DISubroutineType(types: ![[#]], annotations: ![[L4:[0-9]+]])
// CHECK-V2: ![[L4]] = !{![[L5:[0-9]+]]}
// CHECK-V2: ![[L5]] = !{!"btf:type_tag", !"rcu"}
