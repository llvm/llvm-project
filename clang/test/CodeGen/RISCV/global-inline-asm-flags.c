// RUN: %clang_cc1 -triple riscv64 -target-feature +zcmp -flto=full -emit-llvm -o - %s | FileCheck %s
// REQUIRES: riscv-registered-target

asm(".globl func; func: cm.mvsa01 s1, s0; ret");

// CHECK: module asm ".globl func; func: cm.mvsa01 s1, s0; ret"

// CHECK: !{{.*}} = !{i32 6, !"global-asm-symbols", ![[SYM:[0-9]+]]}
// CHECK: ![[SYM]] = !{![[FUNC:[0-9]+]]}
// CHECK: ![[FUNC]] = !{!"func", i32 2050}
// CHECK: !{{.*}} = !{i32 6, !"global-asm-symvers", ![[SYMVERS:[0-9]+]]}
// CHECK: ![[SYMVERS]] = !{}
