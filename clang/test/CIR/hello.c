// Smoke test for ClangIR code generation
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s

void foo() {}
// CHECK: cir.func @foo
