// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s

// External functions to provide side effects that prevent trivial elimination.
void external_f(void);
void external_h(void);

void f(void) { external_f(); }

void h(void) {
  external_h();
  f();
}

// CHECK-LABEL: define{{.*}} void @g()
// CHECK-SAME: [[FLATTEN_ATTR:#[0-9]+]]
__attribute__((flatten))
void g(void) {
  // Flatten recursively inlines: g -> h -> f, so neither call remains.
  // Only the leaf external() call should survive.
  // CHECK-NOT: call {{.*}} @h
  // CHECK-NOT: call {{.*}} @f
  // CHECK: call {{.*}} @external_h
  // CHECK: call {{.*}} @external_f
  h();
}

// CHECK: attributes [[FLATTEN_ATTR]] = {{{.*}}flatten{{.*}}}
