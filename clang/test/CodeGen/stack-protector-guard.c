// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple x86_64-linux-gnu \
// RUN:   -mstack-protector-guard-offset=1024 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple powerpc64le-linux-gnu \
// RUN:   -mstack-protector-guard-offset=1024 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple arm-linux-gnueabi \
// RUN:   -mstack-protector-guard-offset=1024 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple thumbv7-linux-gnueabi \
// RUN:   -mstack-protector-guard-offset=1024 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple aarch64-linux-gnu \
// RUN:   -mstack-protector-guard-offset=1024 -mstack-protector-guard-reg=sp_el0 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=AARCH64
// RUN: %clang_cc1 -mstack-protector-guard=tls -triple riscv64-unknown-elf \
// RUN:   -mstack-protector-guard-offset=44 -mstack-protector-guard-reg=tp \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=RISCV
// RUN: %clang_cc1 -mstack-protector-guard=tls -triple powerpc64-unknown-elf \
// RUN:   -mstack-protector-guard-offset=52 -mstack-protector-guard-reg=r13 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=POWERPC64
// RUN: %clang_cc1 -mstack-protector-guard=tls -triple ppc32-unknown-elf \
// RUN:   -mstack-protector-guard-offset=16 -mstack-protector-guard-reg=r2 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=POWERPC32
void foo(int*);
void bar(int x) {
  int baz[x];
  foo(baz);
}

// CHECK: !llvm.module.flags = !{{{.*}}[[ATTR1:![0-9]+]], [[ATTR2:![0-9]+]]}
// CHECK: [[ATTR1]] = !{i32 1, !"stack-protector-guard", !"sysreg"}
// CHECK: [[ATTR2]] = !{i32 1, !"stack-protector-guard-offset", i32 1024}

// AARCH64: !llvm.module.flags = !{{{.*}}[[ATTR1:![0-9]+]], [[ATTR2:![0-9]+]], [[ATTR3:![0-9]+]]}
// AARCH64: [[ATTR1]] = !{i32 1, !"stack-protector-guard", !"sysreg"}
// AARCH64: [[ATTR2]] = !{i32 1, !"stack-protector-guard-reg", !"sp_el0"}
// AARCH64: [[ATTR3]] = !{i32 1, !"stack-protector-guard-offset", i32 1024}

// RISCV: !llvm.module.flags = !{{{.*}}[[ATTR1:![0-9]+]], [[ATTR2:![0-9]+]], [[ATTR3:![0-9]+]], [[ATTR4:![0-9]+]]}
// RISCV: [[ATTR1]] = !{i32 1, !"stack-protector-guard", !"tls"}
// RISCV: [[ATTR2]] = !{i32 1, !"stack-protector-guard-reg", !"tp"}
// RISCV: [[ATTR3]] = !{i32 1, !"stack-protector-guard-offset", i32 44}

// POWERPC64: !llvm.module.flags = !{{{.*}}[[ATTR1:![0-9]+]], [[ATTR2:![0-9]+]], [[ATTR3:![0-9]+]], [[ATTR4:![0-9]+]]}
// POWERPC64: [[ATTR2]] = !{i32 1, !"stack-protector-guard", !"tls"}
// POWERPC64: [[ATTR3]] = !{i32 1, !"stack-protector-guard-reg", !"r13"}
// POWERPC64: [[ATTR4]] = !{i32 1, !"stack-protector-guard-offset", i32 52}

// POWERPC32: !llvm.module.flags = !{{{.*}}[[ATTR1:![0-9]+]], [[ATTR2:![0-9]+]], [[ATTR3:![0-9]+]], [[ATTR4:![0-9]+]]}
// POWERPC32: [[ATTR2]] = !{i32 1, !"stack-protector-guard", !"tls"}
// POWERPC32: [[ATTR3]] = !{i32 1, !"stack-protector-guard-reg", !"r2"}
// POWERPC32: [[ATTR4]] = !{i32 1, !"stack-protector-guard-offset", i32 16}
