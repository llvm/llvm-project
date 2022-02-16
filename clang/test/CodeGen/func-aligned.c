// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s

// rdar://7270273
void foo(void) __attribute__((aligned (64)));
void foo(void) {
// CHECK: define{{.*}} void @foo() {{.*}} align 64
}
