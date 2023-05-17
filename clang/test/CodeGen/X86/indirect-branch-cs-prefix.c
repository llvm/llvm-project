// RUN: %clang_cc1 -emit-llvm -triple i386 -o - -mindirect-branch-cs-prefix %s | FileCheck %s

// CHECK: !{i32 4, !"indirect_branch_cs_prefix", i32 1}
void foo() {}
