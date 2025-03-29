// Smoke test for ClangIR-to-LLVM IR code generation
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o -  | FileCheck %s

int a;

// CHECK: @a = external dso_local global i32

int b = 2;

// CHECK: @b = dso_local global i32 2
