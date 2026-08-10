// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

// CHECK: @llvm.used = appending global [2 x ptr] [ptr @foo, ptr @X], section "llvm.metadata"
int X __attribute__((used));
int Y;

__attribute__((used)) void foo(void) {}
