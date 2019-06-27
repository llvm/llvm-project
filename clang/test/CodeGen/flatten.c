// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s

void f(void) {}

__attribute__((noinline)) void ni(void) {}

__attribute__((flatten))
// CHECK: define void @g()
void g(void) {
  // CHECK-NOT: call {{.*}} @f
  f();
  // CHECK: call {{.*}} @ni
  ni();
}

void h(void) {
  // CHECK: call {{.*}} @f
  f();
}
