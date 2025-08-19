// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

struct t {
 int (__attribute__((btf_type_tag("rcu"))) *f)();
 int a;
};
int foo(struct t *arg) {
  return arg->a;
}

// CHECK:      !DIDerivedType(tag: DW_TAG_member, name: "f"
// CHECK-SAME: baseType: ![[L18:[0-9]+]]
// CHECK:      ![[L18]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[#]], size: [[#]], annotations: ![[L21:[0-9]+]])
// CHECK:      ![[L21]] = !{![[L22:[0-9]+]]}
// CHECK:      ![[L22]] = !{!"btf_type_tag", !"rcu"}
