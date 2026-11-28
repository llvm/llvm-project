// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s %}

void foo(void) __attribute__((aligned (64)));
void foo(void) {
// CHECK: define{{.*}} void @foo() {{.*}} align 64
}
