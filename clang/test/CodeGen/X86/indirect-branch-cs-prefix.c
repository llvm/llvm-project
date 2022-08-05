// RUN: %clang -target i386-unknown-unknown -o - -emit-llvm -S -mindirect-branch-cs-prefix %s | FileCheck %s

// CHECK: !{i32 4, !"indirect_branch_cs_prefix", i32 1}
void foo() {}
