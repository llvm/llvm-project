// Smoke test for ClangIR-to-LLVM IR code generation
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o -  | FileCheck %s

// TODO: Add checks when proper lowering is implemented.
//       For now, we're just creating an empty module.
// CHECK: ModuleID

void foo() {}
