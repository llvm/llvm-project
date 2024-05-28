// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm -o - | FileCheck %s --check-prefix=GLOBALAS

// CHECK: @llvm.used = appending global [2 x ptr] [ptr @foo, ptr @X], section "llvm.metadata"
// GLOBALAS: @llvm.compiler.used = appending addrspace(1) global [2 x ptr addrspace(1)] [ptr addrspace(1) addrspacecast (ptr @foo to ptr addrspace(1)), ptr addrspace(1) @X], section "llvm.metadata"
int X __attribute__((used));
int Y;

__attribute__((used)) void foo(void) {}
