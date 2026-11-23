// REQUIRES: x86-registered-target
// RUN: %clang -target i386-unknown-unknown -emit-ast -o %t.ast %s
// RUN: %clang -target i386-unknown-unknown -emit-llvm -S -o - %t.ast | FileCheck %s

// CHECK: module asm(target_features: "{{.*}}")
// CHECK-NEXT: "foo"
__asm__("foo");

// CHECK: @g0 = dso_local global i32 0, align 4
int g0;

// CHECK: define dso_local void @f0()
void f0(void) {
}
