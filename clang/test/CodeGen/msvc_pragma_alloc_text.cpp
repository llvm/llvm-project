// RUN: %clang_cc1 -fms-extensions -emit-llvm -o - %s | FileCheck %s

extern "C" {

void foo();
void foo1();
void foo2();
void foo3();

#pragma alloc_text("abc", foo, foo1)
#pragma alloc_text("def", foo2)
#pragma alloc_text("def", foo3)

// CHECK-LABEL: define{{.*}} void @foo() {{.*}} section "abc"
void foo() {}

// CHECK-LABEL: define{{.*}} void @foo1() {{.*}} section "abc"
void foo1() {}

// CHECK-LABEL: define{{.*}} void @foo2() {{.*}} section "def"
void foo2() {}

// CHECK-LABEL: define{{.*}} void @foo3() {{.*}} section "def"
void foo3() {}
}
