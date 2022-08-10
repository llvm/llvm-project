// RUN: %clang_cc1 -triple i386-unknown-unknown -target-feature +sse2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-feature +sse2 -emit-llvm -o - %s | FileCheck %s

// CHECK: define {{.*}}void @_Z3foou6__bf16(bfloat noundef %b)
void foo(__bf16 b) {}
