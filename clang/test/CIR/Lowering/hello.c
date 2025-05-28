// Smoke test for ClangIR-to-LLVM IR code generation
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --input-file %t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=OGCG --input-file %t.ll

int b = 2;

// CHECK: @b = dso_local global i32 2, align 4
// OGCG:  @b = dso_local global i32 2, align 4

int a;

// CHECK: @a = dso_local global i32 0, align 4
// OGCG:  @a = dso_local global i32 0, align 4
