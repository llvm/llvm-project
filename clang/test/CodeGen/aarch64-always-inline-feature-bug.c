// RUN: %clang_cc1 -triple aarch64-- -target-feature +neon -target-feature +sve\
// RUN:   -target-feature -sve -emit-llvm %s -o - | FileCheck %s

// Reproducer for bug where clang would reject always_inline for unrelated
// target features if they were disable with `-feature` on the command line.
// CHECK: @bar
__attribute__((always_inline)) __attribute__((target("neon"))) void foo() {}
void bar() { foo(); }
