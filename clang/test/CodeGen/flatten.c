// UNSUPPORTED: experimental-new-pass-manager
// Currently, different code seems to be intentionally generated under the new
// PM since we alwaysinline functions and not callsites under new PM.
// Under new PM, f() will not be inlined from g() since f is not marked as
// alwaysinline.

// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s

void f(void) {}

__attribute__((noinline)) void ni(void) {}

__attribute__((flatten))
// CHECK: define{{.*}} void @g()
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
