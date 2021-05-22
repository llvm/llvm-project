// RUN: %clang_cc1 -mstack-protector-guard=sysreg \
// RUN:            -mstack-protector-guard-reg=sp_el0 \
// RUN:            -mstack-protector-guard-offset=1024 \
// RUN:            -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-NONE %s
void foo(int*);
void bar(int x) {
  int baz[x];
  foo(baz);
}

// CHECK: !llvm.module.flags = !{{{.*}}[[ATTR1:![0-9]+]], [[ATTR2:![0-9]+]], [[ATTR3:![0-9]+]]}
// CHECK: [[ATTR1]] = !{i32 1, !"stack-protector-guard", !"sysreg"}
// CHECK: [[ATTR2]] = !{i32 1, !"stack-protector-guard-reg", !"sp_el0"}
// CHECK: [[ATTR3]] = !{i32 1, !"stack-protector-guard-offset", i32 1024}
// CHECK-NONE-NOT: !"stack-protector-guard
